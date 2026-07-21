# V88 Learned Dwell Controller: Canonical Design

> Status: implemented; static verification complete, dynamic training validation pending.
>
> Re-read this document before changing the controller, dwell selection, candidate
> probes, operation-route loss, DCT aperture commit order, or related diagnostics.

## 1. Final decision

V88 does **not** introduce an `exit` operation and does **not** use a categorical
`stay / advance / exit` classifier.

The execution contract is:

- all configured refinement steps still execute;
- each step is assigned monotonically to either the current semantic operation or
  the next semantic operation;
- there is no backward jump, arbitrary skip, or discrete early exit;
- update keep and nested ordered contraction continuously shrink an unhelpful
  operation toward the identity boundary;
- learned dwell allocates semantic work; it does not claim wall-clock/GPU savings.

This removes duplicate ownership. A discrete exit head and the continuous
contraction operator would otherwise answer the same question through two competing
control surfaces.

## 2. Why `exit` is removed

The current operator already has a continuous representation of "nothing useful is
left to write":

- update keep can make the residual update approach zero;
- nested contraction can make the active operator approach a smaller ordered
  subspace;
- at their identity boundary, the refinement step remains numerically and
  semantically valid without introducing a separate terminal state.

A learned `exit` classifier would add four avoidable problems:

1. It duplicates the contraction mechanism and creates ambiguous gradient ownership.
2. It turns a continuous marginal-value question into an unstable hard class.
3. Class logits can collapse to a dominant action before the underlying candidate
   values are calibrated.
4. A nominal exit is not real compute saving while candidate probes and the training
   graph still evaluate the remaining path.

Legacy exhaustion/exit code may remain available for historical reproduction, but
it is not part of the V88 main path, loss, command defaults, or trusted diagnostics.

## 3. Current source facts to preserve

The implementation must be based on the refactored policy package, not reconstructed
from old monolithic files.

Relevant ownership currently lives in:

- `clearvla/policy/controller.py`: recurrent control tokens, shared controller
  memory, update/depth controls, and current operation outputs;
- `clearvla/policy/decoder.py`: fixed operation schedule, refinement loop, candidate
  probes, DCT aperture state, and action commits;
- `clearvla/experiments/observed_state_lab/policy_runtime_v39.py`: training losses,
  logging, and legacy dwell helpers;
- `clearvla/cli/train_v40_policy.py`: configuration and command-line exposure.

Known facts that must not be lost:

- the fixed V87 schedule is the neutral behavioral boundary;
- the controller state is multi-token and recurrent, not a pooled scalar state;
- the workspace/controller interface preserves typed evidence roles;
- stage affects low-level evidence selection/query, never the low-level value stream;
- DCT frequency aperture is continuous and monotonic;
- update strength and ordered contraction are continuous transparency coordinates,
  not categorical routing logits;
- action/state values must not leak into evidence value streams through modulation.

The dormant `_dwell_value_targets` helper in the runtime is not the V88 target. It
uses future prefix errors and optional compute cost. Reusing it would silently change
the task from candidate marginal-value estimation into hindsight prefix ranking.

## 4. V88 semantic contract

The monotonic recurrent state is the ordered full-rank refinement block. The six
semantic stage ids are function variants owned by those blocks; they are not six
mandatory recurrent states. At refinement step `s`, only two block assignments are
legal:

- `current`: execute the operation currently owned by the monotonic state;
- `next`: advance once and execute the next operation.

Within each legal block, the value reader may compare the stage variants owned by
that block. At the last block, only its own shelf is legal. The block state is
committed only after a candidate is selected. No candidate may move backward or
skip a block, although its selected function-variant stage id need not be adjacent.

The fixed schedule remains the tie-breaking reference:

- during cold start, always execute the V87 fixed assignment;
- when predicted candidate values are exactly equal, execute the V87 fixed
  assignment for that step;
- zero-initialized V88 parameters must therefore reproduce the fixed V87 forward
  path exactly.

This is a learned dwell mechanism, not random dwell. Random selection is not used as
training supervision or exploration.

## 5. Reader architecture

### 5.1 Input

The value reader consumes the unified controller's recurrent multi-token state and
the shared `ControllerMemory`. It must not reconstruct a second pooled controller.

Every legal candidate receives:

- shared recurrent control tokens;
- a candidate identity token containing semantic stage, current/next relation, and
  refinement step position;
- typed access to the same controller memory used by update/depth control;
- the last committed action and DCT-aperture state through query-side candidate
  context only.

Candidate identity changes **what is read**. It must not be added to evidence values
or rewrite workspace content.

The implemented reader preserves this boundary with three serial query operations:

- a semantic stage/current-next query reads detached global/private
  `ControllerMemory` through cross-attention;
- the resulting horizon queries read detached committed action-frequency tokens,
  annotated by the committed aperture and the shared spectral proposal;
- temporal self-attention composes the 24 horizon queries before the two-channel
  value projection.

The action/aperture lane is a private query context for the value reader. It never
enters workspace evidence values, and every upstream tensor is detached at the
reader boundary.

### 5.2 Output

The reader predicts a candidate marginal-value field with shape conceptually equal
to:

```text
[batch, legal semantic candidates, horizon, 2]
```

The final two channels are:

- arm value, averaged within the arm dimensions at each horizon position;
- continuous gripper value at each horizon position.

This is intentionally not a single scalar. Horizon structure and the distinct arm /
gripper error geometry must survive until the final physical aggregation. The
reader may use typed cross-attention followed by a small output projection; a shared
mean pool followed by one MLP is not sufficient.

Candidate readers share the expensive content-processing weights. Candidate identity
is represented through typed query embeddings and bounded query modulation, not
through separate independent towers that could learn incomparable scales.

### 5.3 What is not a logit

The output is a calibrated estimate of relative residual error, not a class score.
There is no route softmax, route entropy, entropy floor, top-k operation choice, or
hard threshold.

Existing update/depth raw parameters may remain internally parameterized by sigmoid
or another bounded map. They must be logged as continuous controls, not called
operation-route logits.

## 6. Candidate target coordinate

### 6.1 Baseline

At each refinement step, first define a detached baseline from the last committed
state:

- committed action tokens/action state;
- committed semantic stage;
- committed DCT frequency aperture;
- the same physical decoder and action-state normalization used by the main flow
  loss.

The baseline error is measured in physical action space. It is not measured in raw
token RMS, controller-state norm, or unweighted DCT coefficient space.

### 6.2 Candidate probes

Probe every legal current/next candidate from the same detached baseline. A probe
uses its own update keep and ordered contraction, but it must not mutate the
committed state. In V88 phase one the DCT aperture proposal is held common across
candidates. This makes the target identify operator marginal value instead of
confounding operator choice with a second frequency-allocation choice.

Candidate probes run with MMDiT dropout disabled and with CPU/CUDA RNG streams
restored afterward. Enabling the audit therefore cannot alter later main-path
dropout masks. The current workspace read is also held common across candidates;
the target measures operator value at a fixed evidence snapshot.

For candidate `c`, define the per-horizon component error:

```text
E_c[t, arm] = mean_d arm_physical_error(c, t, d)
E_c[t, grip] = gripper_physical_error(c, t)
```

The target field is the candidate's error relative to the committed baseline:

```text
V_c = E_c - E_baseline
```

Negative value means that the candidate explains previously unexplained action
error. Positive value means that it makes the committed estimate worse.

All candidate targets and probe parameters are detached before entering the value
loss. The value loss must not backpropagate through candidate actions, MMDiT blocks,
DCT masks, workspace values, or action encoders.

### 6.3 Candidate-relative centering

Absolute candidate errors are dominated by sample difficulty, diffusion time, and
noise realization. Training directly on them encourages a common-mode predictor.
Therefore center predictions and targets over the legal candidates for each sample,
horizon position, and component:

```text
Vbar_target = V_target - masked_mean_candidates(V_target)
Vbar_pred   = V_pred   - masked_mean_candidates(V_pred)
```

The centered field is the supervised coordinate. Common sample difficulty is removed;
only differences that can change the dwell decision remain.

## 7. Loss design

### 7.1 Primary value loss

Use Smooth L1/Huber regression on the centered candidate-value field:

```text
L_value_raw = huber(Vbar_pred, stopgrad(Vbar_target))
```

Do not use cross-entropy on an argmin label. Hard labels discard the size and spatial
location of the improvement, become noisy near ties, and create the categorical
collapse that this design is intended to avoid.

### 7.2 Physical aggregation

The field loss uses the same horizon/event weighting contract as the physical flow
loss. Arm and gripper must retain the existing per-native-dimension balance. If the
arm channel is already a mean over `arm_dim`, aggregate conceptually as:

```text
L_component = (arm_dim * L_arm + L_gripper) / (arm_dim + 1)
```

This prevents the two-channel representation from accidentally giving one gripper
dimension the same total mass as all arm dimensions combined. Any existing event or
tail weighting is applied once, in this physical aggregation, and not duplicated in
the reader.

### 7.3 Reliability weighting without thresholds

When legal candidates are genuinely indistinguishable, their ordering is mostly
probe noise. Compute a detached target-spread statistic and use a smooth reliability
weight, for example:

```text
spread = rms_candidates(Vbar_target)
reliability = spread / (spread + scale)
L_value = weighted_mean(reliability * L_value_raw)
```

`scale` must come from an EMA or an observed candidate-spread calibration, not an
arbitrary hard cutoff. There is no binary valid/invalid threshold.

The first implementation must log the spread distribution before tuning this scale.

### 7.4 Gradient ownership

The isolated V88 value loss updates only:

- candidate identity/query embeddings;
- typed candidate value reader;
- value output projection.

It does **not** update:

- recurrent controller tokens or shared controller backbone;
- workspace/evidence encoders and value streams;
- action/noisy/stage/low input encoders;
- spectral/DCT aperture reader;
- update keep or nested contraction controls;
- MMDiT action blocks or velocity/event heads.

The selected main action path continues to train all normal model components through
the existing flow and auxiliary losses. This firewall prevents the cheap value head
from changing the candidates it is supposed to audit.

### 7.5 Deliberately absent losses

V88 phase one has no:

- entropy bonus or entropy floor;
- route diversity loss;
- discrete exit loss;
- hard dwell labels;
- compute-cost penalty;
- fixed preferred block-count target;
- loss that forces all candidates to be used equally.

Specialization must arise from candidate-specific inputs and relative value targets,
not from cosmetic distribution regularization.

## 8. Selection and training schedule

### 8.1 Cold start

For the first 200 optimizer steps:

- execute the exact fixed V87 operation schedule;
- still probe legal candidates;
- train only the detached value reader from the value loss;
- log what the learned selector would have chosen, but do not execute it.

The same read-only selector is evaluated in explicit `shadow` mode. Its predicted
spread, fixed-path agreement, and stay/advance decision are real outputs, not
constant placeholders; `operation_decision_shadow_active` distinguishes them from
executed learned dwell.

### 8.2 On-policy dwell

After cold start:

1. predict legal candidate value fields;
2. aggregate each field with the same physical weighting contract;
3. select the candidate with the lowest predicted residual value;
4. use the fixed V87 assignment as deterministic tie fallback;
5. commit exactly one semantic transition and its DCT/action state;
6. continue from the committed state at the next refinement step.

Training and evaluation must both use this on-policy selection after warmup. Training
on the fixed path while evaluating a learned path is forbidden because it changes the
controller state distribution.

The discrete argmin is a control decision and is not differentiated. Learning comes
from value regression; the selected action path receives the ordinary model gradient.

## 9. Relation to nested contraction

Learned dwell and nested contraction have different ownership:

- dwell chooses **which semantic operation** receives the next refinement step;
- update keep chooses **how much residual content** that operation writes;
- nested contraction chooses **how much ordered function capacity** is active.

Maximum contraction/update keep must preserve the original full operation exactly.
Moving inward continuously weakens capacity toward the identity boundary. These
controls must not directly scale the conditioning signal or evidence value amplitude.

No extra exit control is needed: an assigned operation that predicts little useful
work can be continuously reduced by the existing operator controls.

## 10. DCT state and commit order

The DCT aperture is part of the committed recurrent state.

- baseline value uses the **last committed** aperture;
- the controller proposes one aperture update for the refinement step;
- every operation candidate is probed under that same proposal;
- the aperture is committed only with the selected action update;
- no later diagnostic may recompute the baseline under the candidate aperture.

This prevents the value target from attributing a shared coordinate-mask change to
one operation candidate. Candidate-specific aperture routing is intentionally outside
the V88 attribution run.

## 11. Required diagnostics

Remove or retire V88 main-line `exit`, route-softmax, and route-entropy metrics.
Keep legacy names only inside legacy reproduction paths.

V88 must report:

### Value calibration

- target candidate spread, mean and quantiles;
- predicted candidate spread;
- centered Huber value loss;
- predicted/target centered correlation;
- pairwise ordering accuracy for legal candidates;
- fixed-schedule agreement rate;
- common-mode ratio before centering;
- near-tie reliability mean and effective weighted fraction.

### Dwell behavior

- current/next selection rate by refinement step;
- semantic-stage use and dwell length distribution;
- monotonicity violation count, which must be exactly zero by construction;
- fixed tie-fallback count;
- train/eval selector mode and warmup state.

### Operator behavior

- update keep and nested contraction by selected semantic stage;
- selected operation's effective channel/capacity diagnostics;
- action update norm before and after continuous contraction;
- DCT aperture before proposal and after commit.

### Gradient ownership

- value-reader gradient norm;
- shared controller-backbone gradient caused by isolated value loss, expected zero;
- MMDiT/action-block gradient caused by isolated value loss, expected zero;
- ordinary total controller and MMDiT gradients from the selected main path.

Metrics must use scalar reductions before entering the runtime loss-row synchronizer.

## 12. Implementation sequence

1. Delete V88 main-line exit output, selection, loss, and diagnostics while preserving
   explicitly named legacy reproduction paths.
2. Add typed candidate identity tokens and the shared candidate value reader to the
   unified controller package.
3. Refactor candidate probes to return detached per-horizon arm/gripper physical error
   fields from a common committed baseline.
4. Implement centered Huber value loss, smooth reliability weighting, and the gradient
   firewall.
5. Add 200-step fixed execution with shadow predictions.
6. Enable monotonic on-policy current/next selection for both train and eval.
7. Commit action, semantic stage, and DCT aperture atomically.
8. Replace obsolete operation logits/entropy instruments with value calibration and
   dwell diagnostics.

## 13. Static and dynamic acceptance

### Static checks

- no V88 main path consumes an exit probability or exit threshold;
- no operation-route softmax/entropy loss remains;
- candidate identities touch query/selector state only, never evidence values;
- candidate target tensors are detached;
- value-loss parameter allowlist contains only reader/query/output parameters;
- fixed tie fallback calls the actual V87 block schedule, not block zero;
- DCT baseline and candidate commit order are explicit;
- train and eval call the same post-warmup selector.

### Unit tests

- zero reader output reproduces the fixed V87 schedule exactly;
- equal candidate values use the fixed V87 tie fallback;
- refinement block never decreases or skips;
- the last block cannot advance;
- centered targets are invariant to adding a common error field to every candidate;
- value loss produces no gradient outside its allowlist;
- an unselected DCT aperture cannot alter the next baseline;
- scalar metric rows cannot contain vector tensors.

### Golden/smoke checks

- V87 fixed-mode golden remains exact at `atol=0`, `rtol=0`;
- V88 cold-start forward equals V87 fixed forward;
- a short learned-mode smoke produces finite loss and finite gradients;
- train/eval selectors produce identical decisions for identical saved state;
- logged candidate target spread is nonzero before interpreting ranking accuracy.

## 14. Experiment order

Run only after all static and unit checks pass:

1. `V88-shadow`: fixed V87 execution for the whole run, value reader and diagnostics
   active. This validates target learnability without changing the action path.
2. `V88-dwell`: 200-step fixed cold start, then on-policy monotonic dwell.
3. Compare against the same V87 fixed baseline with identical seed, data order,
   checkpoint loading, and optimizer groups.

Do not combine the first attribution run with a compute penalty, a new block count,
an entropy term, or a change to DCT initialization.

## 15. Questions that must be answered from measurements

Only these choices remain intentionally empirical:

- the smooth reliability scale, calibrated from detached target-spread statistics;
- whether arm/gripper fields are sufficient or whether a later near/tail split is
  justified by measured ranking failures;
- the smallest typed reader capacity that preserves candidate-dependent value spread.

These are not reasons to weaken the initial architecture. The first implementation
must preserve the full controller memory interface and horizon-resolved output so a
negative experiment can be attributed to the method rather than a trivial bottleneck.
