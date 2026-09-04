# ClearVLA Schema30 / Schema28-core recovery current issues

Updated: 2026-09-04

This ledger contains only unresolved questions that can change the next source
unit or the release decision. The active graph and invariants live in
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md),
the execution order in
[`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md), and
volatile process state in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
Completed repair narratives belong to Git history or the archive, not here.

## Maintenance contract

Each issue below has one stable question, one latest comparable observation and
one condition for entering source work. At an agreed decision checkpoint, read
the complete available curve once, update these observations in place, select
the highest-priority unresolved behavior and make source the primary evidence.
Do not repeatedly rescan an unchanged log while searching for a different
story.

Replace an observation when a newer checkpoint has the same source/data/
normalizer/decoder contract. Archive the older observation only if it explains
a source decision, contract transition, causal sign reversal or indispensable
baseline. Otherwise the JSONL and Git history are sufficient. Raw rows, full
console excerpts and probe dumps do not belong here.

## Recovery decision (2026-09-03)

The formal source now carries a narrow Schema28-core recovery overlay under a
new Schema30 component ABI. The explicit source fixes are limited to
the training call graph (one formal forward/loss pass), Schema28 S/W/camera and
the profile-owned non-core gripper codec boundary, plus the already-proven
AMP/BF16 cache, cardinality, data/identity and logging infrastructure. Pen keeps
its current-action-state boundary; RDT now uses previous command consistently
through encode/decode/loss/evaluation. The historical Schema30 S/W/camera
semantic delta is not retained. A fresh Pen recovery run was started from
commit `0973f1920cb8467e3b5f048aaaa180e34e27c83f`; no checkpoint was migrated or
resumed, and repaired RDT runs must also start fresh.

## Evidence boundary

- The Pen recovery run has four complete epochs under manifest 30, source
  digest `827268b10af578c61be4ec27581c6fbbd208626a20c080ca0ed820497efa6d9e`
  and the same `32a3a4d7f21f` normalizer as the anchors. Epoch 4 physical
  full/arm/gripper RMSE is `0.07694 / 0.05853 / 0.14453`; bands are
  `0.03174 / 0.05727 / 0.09653`. The run is healthy and continuing, but this
  midpoint is not a release result.
- Schema28 commit `097330a8` remains the latest complete, comparable Pen
  behavior anchor: eight epochs, exact loss ledger, matching split and action
  normalizer. Raw evidence is under
  `new_logs/reference/primary/schema28_action_world_b8_20260831_013140/`.
- Pre-cachefix Schema29 formal runs are invalid behavior evidence: a CUDA BF16
  autocast weight-cache interaction removed formal parameter VJP after detached
  pass0 even though losses, activations and optimizer steps stayed finite. The
  `d8a77a1` cachefix Pen run passed the real VJP gate and now supplies a complete
  eight-epoch rejected-self-conditioning comparator. It is not the recovery
  graph, and the current tree retains only its proven lifecycle repair.
- The current RDT-8 log is a Schema29/cachefix adapter run, not a recovery-core
  counterpart. Its per-task rows may judge the multitask interface, but cannot
  be merged with the Pen recovery result or used to credit this core repair.
- Smoke, checkpoint round-trip and read-only validation establish interface
  correctness, not learned behavior. Low-coverage interventions establish only
  the boundary they actually changed.
- The active CALVIN run is a fresh initialization from commit `f9cee96a` with
  `arm_flow_mode=relative_command_direct`; its two arm-field branches encode
  the same native relative TCP command. The real CUDA smoke and preflight pass,
  both loss ledgers close, continuous-gripper contribution and dynamic
  conditioning are exactly zero, and all named owners have finite gradients.
  No validation epoch exists yet, so these facts establish the repaired
  interface but say nothing about learned behavior.

## Current decision order

| Priority | Unresolved behavior | Latest compact observation | What selects source work |
|---|---|---|---|
| P0 | complete recovery decision | E4 beats matched Schema28/29 on every major physical RMSE row, with no hard health failure | the complete E8 curve and formal recovery gate |
| P0 | far horizon and continuous gripper | E4 far and gripper RMSE are improved, but decoded recall is `0.3324` and branch disagreement grows with horizon | stability through E8 without a near/arm trade |
| P0 | CALVIN action-outlet closure | the future-gripper shortcut is isolated and the relative TCP arm command now uses two direct branches; smoke/preflight pass but no fresh validation exists yet | a fresh complete curve, read-only causal replay and `open_drawer` closed-loop success |
| P1 | outer action/world self-consistency | the Schema29-scale regression is removed; E4 residuals remain 20-28% above matched Schema28 E4 | persistence or renewed growth on the complete curve |
| P1 | S/W typed chronology | public/action amplitude exists, while typed interval separation and W/Teacher amplitude ratios remain weak | behavior or a matched intervention ties this under-use to far/gripper error |
| P1 | geometry address responsibility | recovery geometry address remains almost action-neutral; geometry value utility is not stable at current coverage | repeated matched evidence with adequate coverage, preferably on Pen and RDT |
| P2 | bottom typed-P2 selection | five routes remain almost exactly uniform from Schema25 through Schema30 although the value update is nonzero | prove that replacing the equal mixture changes deployed action utility |

Finite threshold crossings, early semantic-far sign changes and execution
capacity are secondary context. They do not outrank the decisions above.

## S30-01 — historical self-conditioning observation; recovery removes train-time pass0

The historical Schema30 training path estimated one detached clean endpoint,
rebuilt W, and sent the loss through a second pass. The recovery tree removes
that train-time pass0/pass1 lifecycle: it performs one formal velocity forward
and one loss composition. Deployment is unchanged and still performs one
complete proposal ODE, rebuilds W once, then performs one complete refined ODE.
The final action may move away from the action that conditioned W.

Schema28 showed a real correction and a remaining residual:

```text
proposal -> refined action RMS          0.02514
final interval vs W-condition mismatch 0.02933
final adjacent-delta mismatch           0.01514
```

Open question: does recovery retain its far-horizon/gripper improvement and
keep the final mismatch from growing through E8? Close only with the complete
validation curve and matched proposal/refinement measurements. A third
unconsumed `W(final)` call would not close this issue.

The rejected Schema29 comparator reached
`0.17451 / 0.19102 / 0.07348` at epoch 4 for proposal-to-refined change,
final interval mismatch and final adjacent-delta mismatch. Recovery epoch 4 is
`0.01921 / 0.02297 / 0.01117`, a reduction of roughly 85-89%. Matched
Schema28 epoch 4 is `0.01495 / 0.01798 / 0.00934`, so recovery is still about
20-28% higher at the same training age. The main regression is recovered; keep
this issue open only as an E8 guard, not as authorization for another W or ODE
pass.

## S30-02 — far horizon and continuous gripper remain the behavior target

Schema28 final validation was:

```text
full / arm / gripper physical RMSE  0.07657 / 0.05677 / 0.14733
horizon bands 1-4 / 5-12 / 13-24   0.02502 / 0.05513 / 0.09743
tail / first                         7.66
decoded gripper P / R / F1           0.6006 / 0.2749 / 0.3771
decoded predicted / target events    621 / 1357
```

The low event recall and far-gripper error show a deployed continuous-action
failure, not merely an aggregate-scale artifact. The recovery graph restores
the Schema28 operands for Pen, whose qpos and action-command gripper values are
identical. RDT is different: previous command owns the gripper codec boundary
while qpos remains separately observed and still anchors arm/network physical
conditioning. Never trade a better near band for a worse far band or use an
event-head accuracy as a substitute for decoded gripper behavior.

At matched epoch 4, recovery physical full/arm/gripper RMSE is
`0.07694 / 0.05853 / 0.14453`, versus Schema28
`0.07951 / 0.06114 / 0.14775` and Schema29
`0.08046 / 0.06261 / 0.14764`. Its bands
`0.03174 / 0.05727 / 0.09653` are also best at that checkpoint. Relative to
Schema28 final E8, recovery E4 already has slightly better gripper and far-band
RMSE, while arm, first and near-band rows still have plausible training room.
Decoded gripper P/R/F1 is `0.5666 / 0.3324 / 0.4190`, with `796 / 1357`
predicted/target events. This is a real midpoint recovery, not closure; require
the complete curve to show that it is retained.

## S30-03 — S/W chronology is active but weakly differentiated

At Schema28 final validation, W2 prediction-to-Teacher amplitude was about
`0.69x` for semantics and `0.44x` for transport. Both W and P2 consumers had
nonzero gradients, so this was under-use rather than a proven disconnection.
Historical Schema30 gave typed W innovation both learned chronology and the
physical action condition. Recovery restores the Schema28 action-carrier
modulation; no new chronology hypothesis is being promoted.

Open question: does that semantic repair produce distinct late intervals and
useful transport on validation? Read every interval's prediction/Teacher
semantic and transport rows together with action ablations. Do not change a
loss weight, add a quota, or amplify transport merely because one RMS is small.

At recovery epoch 4 validation, S public interval variation is `0.16146`, while
semantic/appearance/geometry typed variation is
`0.01501 / 0.01296 / 0.01009`. W prediction interval variation is `0.06846`
against Teacher `0.12300`. W2 semantic/Teacher amplitude is
`0.20099 / 0.32841` (about `0.61x`) and transport is
`0.03369 / 0.08684` (about `0.39x`). This remains under-use, but its normalized
gap is not worse than the comparable anchor and the complete consequence chain
has strong matched far/gripper responsibility. There is no current evidence of
a disconnected W path or a reason to alter its amplitude.

## S30-04 — bottom typed-P2 carries values but does not select routes

This is the V120-bottom router, not object-P2 semantic/geometry interval
selection. Across the last available windows of Schema25 R1/R2, Schema26,
Schema27 and the later Schema28-30 runs, its five source masses remain about
`0.2` each, normalized entropy remains approximately `1`, and query-axis route
variation remains exactly `0`. Schema27's largest route mass was only about
`0.201`; Schema28 was still effectively uniform after eight epochs.

The routed update is nonzero and can grow, so this is not a dead path. The
current interpretation is equal-mixture use without typed/query specialization.
Do not add entropy pressure, quotas or forced specialization. Promote it to a
source unit only after an action-level intervention shows that the missing
selection, rather than the routed values themselves, limits behavior.

## S30-05 — geometry address remains weak; recovery value utility is unconfirmed

Schema28 matched interventions found:

```text
geometry address neutral: far arm/gripper action delta  0.00050 / 0.00106
geometry value zero:      far-gripper action delta       about 0.00729
far semantic zero:        far arm/gripper action delta   0.02642 / 0.11407
```

Historical Schema30 corrected camera validity ownership and treated camera
support as width metadata. Recovery restores the Schema28
validity-times-support coordinate/transport reduction while preserving
independent semantic-K and geometry-K*C selection.
Open question: does geometry gain responsibility on Pen or the richer RDT-8
camera/task distribution? A weak but correctly connected geometry lane is not
automatically a defect; deletion, forced mass and learned gain remain
unsupported.

On the 8.94%-coverage recovery epoch-4 panel, neutralizing geometry address
changes far-gripper action by only `0.00106` RMS and is MSE-neutral. Zeroing
geometry value changes it by `0.00645`, but removal slightly improves MSE by
`0.00028`; this does not reproduce the historical positive geometry-value
evidence strongly enough to assign stable utility. By contrast, zeroing far
semantic value changes far-gripper action by `0.11325` and worsens MSE by
`0.02307`, while neutralizing the full consequence chain changes it by
`0.13280` and worsens MSE by `0.03105`. Geometry is connected but weak; semantic
consequence is the presently demonstrated owner. Do not modify geometry before
E8 or without stronger cross-outlet evidence.

## Secondary observations

- Finite threshold crossings are retrospective telemetry only. Revisit them
  only when coupled reproducibly to sustained non-recovery, parameter/optimizer
  damage, validation regression or another hard-stop failure.
- Execution capacity remains nearly full and hard terminal occupancy remains
  zero. This is not a release failure unless a matched behavior/cost decision
  demonstrates that the controller's lack of selection matters.

## S30-06 — RDT-8 currently validates the adapter, not native multiview/bimanual support

The first multitask outlet uses eight-task balanced sampling, high + right-wrist
cameras, and a right-arm 7-D projection from native 14-D action. Task identity
controls sampling, validation and logging only. It is not a hidden model input.

The current Schema29/cachefix RDT-8 run has four complete epochs. Its aggregate
epoch-4 physical full/arm/gripper RMSE is `0.2324 / 0.0935 / 0.5708`, with
bands `0.1265 / 0.2113 / 0.2701`; every task contributes 64 validation samples.
This is useful adapter/interface evidence but not recovery-core evidence.

Open question: are all eight task rows healthy under one shared core and are
failures task/profile-specific rather than global? Report every task with its
sample/event counts and action/camera profile. Native three-camera consumption,
depth and 14-D bimanual modeling remain later explicit ABI units; the adapter
must not be described as already solving them.

## S30-07 — B-spine is a conditional bottom representation question

The active bottom already receives the complete `x_t [B,24,18]` physical field,
but its `NativeTimePhysicalActionTokenLift` is a per-row component projection
followed by `LayerNorm -> MLP`. Source review therefore supports testing a
parallel fixed temporal chart, not replacing the lift or changing the flow
field. B-spline Policy and Spline Policy motivate local-support control points,
continuous resampling and local correction; they do not establish that an
internal spine should become the action output or add an inference loop.

The open behavior decision is now Pen-scoped: whether the fixed B-spline
coarse/detail view reaches the bottom action path and preserves far-horizon,
gripper-event and detail information without a near/arm trade. Gate A selected
one cubic `T=24/K=12` chart from the shared Pen/RDT dataset traversal; that RDT
chart evidence is not a request to start an RDT training job. Gate B is the
zero-initialized, bias-free bottom-only branch beside `noisy_lift`, fused under
the existing `action_state_factor`. Its Schema31 config/manifest identity is
fixed, and the combined local forward/reverse/checkpoint/RNG/optimizer/runtime
gate is closed. Real CUDA/BF16 owner-VJP, one Pen B8 smoke and read-only
checkpoint validation remain before Gate D starts one fresh Pen run with the
matched `spine_zero` evaluation intervention. RDT and CALVIN are explicitly
outside this first training experiment.

Reject the proposal if the chart is rank-deficient/non-finite, strongly
amplifying, the learned view stays unused, or it improves only by suppressing
far/event detail. Do not rescue it with a gain,
clip, quota, entropy target, extra loss or an additional ODE/W pass. The active
field codec is `clearvla/mainline/model/action_codec.py`; legacy DCTFlowCodec and
historical output spline heads are comparison evidence only.

## S30-08 — CALVIN needs a direct relative-command arm chart and an isolated binary command head

The first `open_drawer` formal run reached epoch 2 with arm physical RMSE
`0.12726`, command F1 `0.24713`, predicted-positive rate `0.65671` against
target `0.13174`, and decoded event ratio `12.54375`. Those values alone did
not identify the cause. A fixed-checkpoint intervention did: the endpoint head
scored the complete target field at accuracy/F1 `1/1`, replacing only the six
continuous gripper coordinates flipped `54.17%` of commands, and changing only
their initial ODE noise flipped `46.35%`; changing all arm noise flipped only
`8.33%`. The continuous gripper field itself changed by exactly zero across
the five ODE updates. Training therefore exposed future targets at a boundary
where deployment supplied persistent random source noise.

The accepted outlet repair preserves the complete 18-D flow and audit ABI but
zeros those six coordinates before any dynamic P/transition/bottom consumer in
CALVIN binary mode. Pen/RDT receive the original tensor unchanged. The command
CE retains weight `.1` and equalizes the two states by their horizon-weighted
mass only when both occur in the batch; it does not change the dataset sampler
or invent a missing class in single-class batches. The smoke checkpoint passes
endpoint and sampled initial-gripper-noise invariance with exact-zero command,
logit and arm-action deltas, and its read-only replay reports zero source delta
with optimizer/scheduler/RNG loads and checkpoint writes disabled.

The next completed eight-epoch run exposed a separate arm-chart mismatch. A
CALVIN row is already a six-dimensional relative world-TCP command. The old
`legacy_independent` codec treated it as an absolute arm pose in branch 0 and
an adjacent command difference in branch 1, then cumulatively integrated
branch 1 at decode. Its train arm flow fell by `71.1%`, but validation arm RMSE
stayed near `0.11-0.13` and ended at `0.1121`; this train/validation decoupling
was expected from the incompatible chart, not evidence that the direct command
itself was intrinsically hard.

The accepted arm repair is also outlet-scoped and keeps the 18-D ABI: both arm
branches encode the same relative command, decode remains the established
`0.75/0.25` branch blend, branch consistency remains supervised, and temporal
difference/cumulative integration/adjacent-command smoothness are disabled.
Pen/RDT retain their historical absolute-plus-delta chart. Focused CPU tests,
Ruff, compileall and the real BF16 CUDA smoke pass; the formal run uses commit
`f9cee96a`, source digest
`6e4dd12d43d1f63ad090190d167531c53ecd385d5eb2a355b0692a7fef22e797`
and the same CALVIN split/normalizer as the rejected comparator.

Both structural defects are closed, but behavior is not. Keep this issue open
until the fresh eight-epoch direct-arm run reports source-native and normalized
arm RMSE by horizon, command precision/recall/F1, predicted and target state
rates, balanced/unbalanced CE, direct-branch disagreement, and the exact-zero
continuous-gripper conditioning/contribution together. Then repeat the
read-only causal probe and run the real CALVIN `open_drawer` closed-loop
benchmark. The two old runs and the smoke checkpoint are causal/interface
evidence only and must never be resumed into this ABI.

## Closed facts that should not be reopened without contradictory evidence

- W and ControlledTransition have independent matched action responsibility;
  W remains the only world producer and CT remains a transition consumer.
- Semantic P2 carries strong far/gripper responsibility. Geometry being weaker
  does not invalidate that result.
- The Schema29 autocast-cache VJP break is closed by local cache isolation and
  a real CUDA parameter-owner gate.
- Schema30's P3 source-depth compiler, data/identity infrastructure and the
  cache/VJP repair remain accepted. Its disputed S fusion, W chronology,
  camera-support semantics and gripper boundary are historical only; the local
  recovery source restores Schema28 behavior for those owners.
- Capacity near full width is not a release failure or a hardware-rank claim.

## Stop and closure rules

Hard-stop a run for non-finite values, identity/lineage failure, an open loss
ledger, vanished formal parameter-owner VJP, checkpoint ABI violation, process
memory above 22 GiB. Finite threshold crossings are secondary telemetry and do
not stop a run by themselves; act only on a demonstrated persistent health or
behavior failure associated with them. Early event F1, small geometry RMS or
warmup capacity likewise does not stop alone.

Close an issue only when the producer-to-consumer forward path, the
loss-to-owner-to-optimizer reverse path and the relevant matched intervention
all agree, with stated coverage. A source explanation without behavior evidence
remains open; an unusual metric without attributed responsibility does not
authorize a structural edit.
