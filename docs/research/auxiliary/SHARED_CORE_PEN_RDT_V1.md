# Shared Pen/RDT core candidate v1

This is a fresh candidate based on the completed Schema28 recovery source
(`0973f192`). It is not a replacement for the recovery behavior anchor and it
does not reuse a trained checkpoint.

## Shared graph

Pen and RDT use the same G/S/W/P, P2/P3, controlled-transition and V120
execution graph. The configs differ only at the outlet/data boundary:

```text
language_conditioning_mode = task_anchor_v1
action_frame_weight_mode   = event_motion_v1
event_gain                 = 0.75
motion_gain                = 0.35
```

The language task anchor is zero-initialized and S-owned. It steers the
interval goal query, never enters W directly and cannot create object support.
Frame weights are derived from target-only event/motion masks, retain every
row, and are renormalized to the existing horizon-weighted mass.

RDT now carries its previous-command gripper boundary through online history,
codec encode/decode, training loss, validation and deployment sampling. Pen
falls back to its current action-state boundary when legacy callers omit the
optional field.

The active action-DiT FFN width is explicitly wired from
`BottomConfig.ffn_expansion` to `latent_cvae_ffn_expansion`; the decoder and
compatibility wrapper no longer have a silent configuration split.

## Configs and launchers

- `configs/mainline/object_intent_dynamics_323_pen_shared_v1.json`
- `configs/mainline/object_intent_dynamics_323_rdt_shared_v1.json`
- `scripts/train_pen_shared_core.sh`
- `scripts/train_rdt_shared_core.sh`

Both configs keep the recovery optimizer, uniform E5 deployment schedule,
codec dimensions, loss coefficients and five-step lifecycle. RDT retains its
profile-owned threshold `0.18310546875`, task manifest and previous-command
boundary.

## Acceptance

Before any formal run, require config/identity checks, real CUDA/BF16 owner-VJP,
B8 smoke, read-only checkpoint validation, and complete per-task or Pen
behavior curves. Acceptance must report language anchor RMS/gradients, frame
weight range and weighted fractions, RDT boundary closure, all horizon bands,
decoded gripper event P/R/F1, and the exact loss ledger. No early loss or
internal language/weight activation is a behavior claim.

## Pen trajectory frame-weight scope control

The 2026-09-11 preset
`object_intent_dynamics_323_pen_gripper_frame_scope_v1.json` differs from the
Pen shared preset only in output path and
`objectives.gripper_trajectory_weight_mode=horizon_only`. The `.03` gripper
trajectory budget and deployment-cumulative operands stay fixed. Frame weights
still apply to flow, decoded action, motion, smooth-delta and physical-delta
consistency. Separately normalized transition/persistence means now omit their
within-owner event/motion multiplier. This primarily tests cross-task-channel
motion reweighting, not dropping events or reallocating the two owner budgets.

Acceptance compares every matched epoch against `20260908-pen-shared-core`:
full/arm/gripper and three horizon bands, decoded event P/R/F1/counts,
post-event 1-2/3-6/7+, actual cumulative-branch error and branch disagreement.
Do not claim success from a lower auxiliary loss alone. Log the mode code in
`metrics.jsonl` and the objective console row. No acceleration, B-spline,
geometry gain, new solver or other integration-line change is included.
