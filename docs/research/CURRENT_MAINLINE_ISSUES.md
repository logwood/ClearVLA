# ClearVLA Schema30 current issues

Updated: 2026-09-02

This ledger contains only unresolved questions that can change the next source
unit or the release decision. The active graph and invariants live in
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md),
the execution order in
[`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md), and
volatile process state in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
Completed repair narratives belong to Git history or the archive, not here.

## Evidence boundary

- Schema30 (`object_intent_dynamics_323`, manifest 30) is the active candidate.
  Fresh Pen and RDT-8 formal runs are active, but neither had produced a
  validation epoch at the latest handoff snapshot. Early train rows are health
  evidence only.
- Schema28 commit `097330a8` remains the latest complete, comparable Pen
  behavior anchor: eight epochs, exact loss ledger, matching split and action
  normalizer. Raw evidence is under
  `new_logs/reference/primary/schema28_action_world_b8_20260831_013140/`.
- Schema29 formal runs are invalid behavior evidence. A CUDA BF16 autocast
  weight-cache interaction removed formal parameter VJP after the detached
  pass0 even though losses, activations and optimizer steps stayed finite.
  Commit `d8a77a1` repaired that lifecycle; Schema30 retains the repair and has
  passed the real Pen B8 VJP gate.
- Smoke, checkpoint round-trip and read-only validation establish interface
  correctness, not learned behavior. Low-coverage interventions establish only
  the boundary they actually changed.

## S30-01 — self-conditioning closes the training call graph, not the final fixed point

Training now estimates one detached clean endpoint from the same noisy field,
rebuilds only W, and applies the sole action/future loss to the formal second
pass. Deployment still performs one complete proposal ODE, rebuilds W once,
then performs one complete refined ODE. The final action may move away from the
action that conditioned W.

Schema28 showed a real correction and a remaining residual:

```text
proposal -> refined action RMS          0.02514
final interval vs W-condition mismatch 0.02933
final adjacent-delta mismatch           0.01514
```

Open question: does Schema30 improve far-horizon/gripper behavior and keep the
final mismatch from growing? Close only with a complete validation curve and
matched proposal/refinement measurements. A third unconsumed `W(final)` call
would not close this issue.

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
failure, not merely an aggregate-scale artifact. Schema30 repaired the
trajectory-supervision operands to match the deployed absolute and cumulative
delta codec, but behavior remains open until validation. Never trade a better
near band for a worse far band or use event-head accuracy as a substitute for
decoded gripper behavior.

## S30-03 — temporal/transport bandwidth is active but still unproven

At Schema28 final validation, W2 prediction-to-Teacher amplitude was about
`0.69x` for semantics and `0.44x` for transport. Both W and P2 consumers had
nonzero gradients, so this was under-use rather than a proven disconnection.
Schema30 gives typed W innovation both learned chronology and the physical
action condition; zero action no longer erases interval identity.

Open question: does that semantic repair produce distinct late intervals and
useful transport on validation? Read every interval's prediction/Teacher
semantic and transport rows together with action ablations. Do not change a
loss weight, add a quota, or amplify transport merely because one RMS is small.

## S30-04 — geometry has a legal path but weak established action responsibility

Schema28 matched interventions found:

```text
geometry address neutral: far arm/gripper action delta  0.00050 / 0.00106
geometry value zero:      far-gripper action delta       about 0.00729
far semantic zero:        far arm/gripper action delta   0.02642 / 0.11407
```

Schema30 corrected camera validity ownership and keeps camera support as width
metadata, while preserving independent semantic-K and geometry-K*C selection.
Open question: does geometry gain responsibility on Pen or the richer RDT-8
camera/task distribution? A weak but correctly connected geometry lane is not
automatically a defect; deletion, forced mass and learned gain remain
unsupported.

## S30-05 — finite spikes are a watch item, not yet one proven mechanism

Schema28 had 12 finite threshold-5 crossings, mainly observation flow/
`target_dino_key` plus three output-head events; maximum global preclip was
`24.63`. Schema30's early formal rows are finite and ledger-closed, but the
current runs have already shown:

- seven early Pen crossings through step 106, dominated by
  `bottom.decoder.velocity_head.arm_abs.weight`, with no further Pen crossing
  through the latest completed window at step 2000;
- three early RDT output-head crossings followed by three separated
  observation-side crossings at steps 954/1251/1317. The maximum was step 954:
  `target_dino_key.1.weight` owner L2 `19.28`, global preclip `22.60`;
  later owners were `target_dino_key` and `flow.delta_head`.

One finite recurrence does not establish the proposed “G1/G2 address gradient
confluence” explanation. Keep training unless crossings become persistent,
grow materially, produce non-finite values, or coincide with a behavior/
representation break. If that happens, replay the same checkpoint/batch with
per-loss parameter VJP before changing clipping, LR, normalization or loss
weights.

## S30-06 — RDT-8 currently validates the adapter, not native multiview/bimanual support

The first multitask outlet uses eight-task balanced sampling, high + right-wrist
cameras, and a right-arm 7-D projection from native 14-D action. Task identity
controls sampling, validation and logging only. It is not a hidden model input.

Open question: are all eight task rows healthy under one shared core and are
failures task/profile-specific rather than global? Report every task with its
sample/event counts and action/camera profile. Native three-camera consumption,
depth and 14-D bimanual modeling remain later explicit ABI units; the adapter
must not be described as already solving them.

## Closed facts that should not be reopened without contradictory evidence

- W and ControlledTransition have independent matched action responsibility;
  W remains the only world producer and CT remains a transition consumer.
- Semantic P2 carries strong far/gripper responsibility. Geometry being weaker
  does not invalidate that result.
- The Schema29 autocast-cache VJP break is closed by local cache isolation and
  a real CUDA parameter-owner gate.
- Schema30's S fusion, W chronology, single-validity boundary, camera-support
  semantics, gripper codec operands and P3 source-depth compiler passed forward,
  reverse, checkpoint and smoke gates without new parameters, losses or passes.
- Capacity near full width is not a release failure or a hardware-rank claim.

## Stop and closure rules

Hard-stop a run for non-finite values, identity/lineage failure, an open loss
ledger, vanished formal parameter-owner VJP, checkpoint ABI violation, process
memory above 22 GiB, or a persistent severe spike pattern. Early event F1,
small geometry RMS, one finite spike or warmup capacity does not stop alone.

Close an issue only when the producer-to-consumer forward path, the
loss-to-owner-to-optimizer reverse path and the relevant matched intervention
all agree, with stated coverage. A source explanation without behavior evidence
remains open; an unusual metric without attributed responsibility does not
authorize a structural edit.
