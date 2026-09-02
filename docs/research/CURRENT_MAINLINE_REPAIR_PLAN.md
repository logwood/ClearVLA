# ClearVLA Schema30 execution and closure plan

Status: formal Pen and RDT-8 runs are active on one Schema30 source identity.
Updated: 2026-09-02

This plan contains only the remaining execution sequence. Architecture truth is
in [`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md);
open questions are in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md); live PIDs, run tags
and last observed steps are in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
The completed Schema25-to-Schema30 implementation narrative is recoverable from
Git commit `f60bd80` and the replay archive.

## Objective

Use one core source commit and two experimental outlets:

| Outlet | What it decides | What it does not decide |
|---|---|---|
| Pen single-task | far horizon, continuous gripper, W/P2/consequence/CT closure, final refinement mismatch, spike ecology | cross-task or richer camera/action adapter behavior |
| RDT-8 multitask | dataset/language/camera/action ABI, balanced sampling, per-task behavior and shared-core competition | native three-camera, depth or full 14-D bimanual modeling |

The two runs may differ only in dataset/adapter/task profile and declared
normalizers. They must serialize the same capability, Schema30 manifest, source
digest, optimizer ownership and loss ledger.

## Current phase: observe, do not edit the graph

The source, real CUDA VJP, Pen/RDT smoke and read-only checkpoint validations
have passed. Both formal runs started from empty directories. Unless a hard-stop
condition occurs, let them continue; do not alter structure in response to one
early metric magnitude or one finite spike.

Use the following review cadence:

1. inspect compact health windows during epoch 1 for identity, ledger,
   non-finite values, raw owner gradients, memory and spike ownership;
2. make the first behavior comparison only after a complete validation row;
3. review Pen again around epoch 2, epoch 4 and epoch 8; retain all completed
   epochs rather than selecting the best one;
4. review RDT-8 by task at its first validation and each later completed epoch;
   macro and micro summaries never replace the eight task rows;
5. run matched causal probes only when the checkpoint and the unresolved
   decision require them. Do not add routine probe cost to every train step.

The canonical audit command is:

```bash
python -m clearvla.tools.audit_policy_logs runs/<run-tag> --format text
```

For a release comparison, pass the complete run directory together with the
locally available V120/Schema28 anchors. Verify split, action normalizer,
decoder, batch size and completed epochs before calling a delta architectural.

## Health gate at every checkpoint

Continue only when all of the following remain true:

- manifest/config/source and dataset identities match the run context;
- `loss_ledger_gap` and contribution gap remain numerical zero;
- formal parameter-owner gradients remain present on active paths;
- G mass conservation, P2 no-null terminal, CandidateWorld tag identity and
  capacity non-expansiveness remain valid;
- CUDA values remain finite and process memory stays at or below 22 GiB;
- a spike is finite, attributable and non-persistent rather than an escalating
  multi-owner pattern;
- checkpoints are atomic and remain exact-resume compatible only with the same
  source identity.

Hard-stop only for non-finite values, lineage failure, an open ledger, vanished
formal VJP, checkpoint ABI failure, memory overflow or persistent severe
spikes. Early event F1, small geometry RMS, full capacity or a single finite
crossing are review signals, not independent stop rules.

## Pen decision surface

Compare every completed validation epoch with the complete Schema28 anchor on:

- physical and normalized full, first, first-8, tail and three horizon bands;
- arm and gripper RMSE;
- decoded gripper precision/recall/F1, predicted/target event counts and
  post-event horizons;
- W prediction versus Teacher semantic/transport variation for every interval;
- proposal-to-refined action change and final action-to-W condition mismatch;
- G/S/W/P1/P2/P3/CT/bottom owner gradients and exact loss contributions;
- spike count, maximum and owner after exposure normalization.

Schema30 passes the core behavior gate only if improvement is not bought by a
near/far or arm/gripper trade and the final result, not merely the best epoch,
remains competitive.

## RDT-8 decision surface

For each of the eight tasks report:

- train/validation sample count and sampler mass;
- camera and action profile plus normalizer/language identity;
- full, arm-or-joint-group, gripper and horizon metrics;
- decoded event counts and validity coverage;
- relevant core owner gradients and any task-local spike or non-finite event.

First classify a failure before touching the core:

| Pen | RDT-8 | Primary interpretation |
|---|---|---|
| bad | bad across tasks | shared core or training engine |
| healthy/improved | broadly bad | adapter, profile, sampler or task competition |
| healthy/improved | most healthy, a few bad | task-local data/action/camera semantics |
| bad | mixed/healthy | Pen behavior contract or task-specific core demand |

Stop only the affected outlet when identity and evidence localize the failure
outside the shared core.

## How the next source unit is chosen

After a decision checkpoint:

1. state the behavior failure without proposing a mechanism;
2. trace its exact producer -> transform -> consumer path in current source;
3. trace loss -> tensor VJP -> parameter owner -> optimizer in reverse;
4. use an existing matched intervention, or add the smallest validation-only
   one whose result would select between competing explanations;
5. change one coherent semantic unit, with explicit axes, zero semantics,
   dtype, scale and call frequency;
6. re-run forward/reverse, checkpoint, CPU/static, real CUDA VJP and both smoke
   gates before the next formal pair.

Do not bundle an amplitude gain, quota, hard event gate, entropy target, extra
clip or loss-weight change with a connection repair. Such changes make the
result harder to attribute and risk numerical hardening.

## Conditional next phase

If Schema30 has no new core failure and its behavior is at least stable, the
next design review may consider a B-spine numerical trajectory view: a
top-level, bounded representation that lets the model reason over coherent
long-horizon action geometry while the bottom retains its existing local
numerical competence. This is not yet an active source contract.

The older B-spline-output and Minimal Iterative Policy proposals remain
historical. MIP returns only if a source-grounded review identifies a distinct
consumer and shows that a bounded correction adds information beyond the
existing two-pass outer refinement. Neither proposal should be merged into a
Schema30 diagnosis by default.

## Completion

Schema30 is behavior-complete only after both outlets finish their required
curves, all hard gates remain closed, Pen is judged against comparable anchors,
and RDT-8 is judged per task. At that point update the architecture contract
with the accepted decision, move superseded detail to the archive, and retain
only the evidence needed to select the next semantic unit.
