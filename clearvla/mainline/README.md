# ClearVLA independent mainline

`clearvla/mainline/` is the only active capability-named implementation. The
historical V39 CLI/runtime/trunk and V-numbered launchers are migration or
comparison references, not runtime parents of this package.

The compact architecture source of truth is
[`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`](../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md).

## Active vertical path

```text
config / manifest / typed online and training inputs
  -> three-frame current observation + two learned flows
  -> Pre-G local chart
  -> G1/G2/G3 progressive local grounding with an N=49 rematerialization
  -> one dense global K+null grounding objective
  -> S public interval carrier + per-type [interval,K,type] relevance
  -> typed-free coarse physical proposal [B,4,7]
  -> goal-invariant W(ObjectWorldBelief, PhysicalActionCondition)
  -> action-tagged four-interval CandidateWorld
  -> P1 one cached protected-detail read over all progressive candidates
  -> per-ODE V120 P1 policy block
  -> P2 bounded consequence read
  -> P3 temporal/state-change optional lanes plus protected carriers
  -> shared V120 action/context canvas seed
  -> true P1/P2 terminal layer contracts
  -> per-ODE noisy-action controlled transition
  -> V120 Evidence MMDiT / execution value / capacity bottom
```

Training adds one no-grad future-teacher branch. Future evidence has a separate
typed input and cannot enter the online/deployment API.

The 24-row executed-history future proposal is an auxiliary supervised
prediction in the recovered V120 object path. Its separate 4-recent +
3-summary history encoding remains the observable executed-history condition
used by the shared V120 canvas seed; it is not a P1 or controlled-action alias.

## Package boundaries

```text
clearvla/mainline/
  config.py          active nested configuration
  manifest.py        compact architecture identity
  interfaces.py      disjoint online/training inputs
  model/             observation, G/S/W/P, transition and bottom
  training/          objectives, optimizer and engine
  runtime/           sampling, evaluation, logs and checkpoints
  train.py           direct entry point
```

This package must not import the legacy trainer/runtime/trunk or a
`current_vXXX` launcher. Audited low-level numerical primitives are extracted
under `v120_core/`; their callers still own explicit typed boundaries.

## Change policy

The recovery reference is V120 `long`, commit
`0b92d359a2889a0a1b1eba256007c00ccbc54f3c`.

- Preserve active V120 tensor semantics, source ordering, residual positions,
  dynamic frequency and loss ownership.
- Repair a behavior only when source semantics are self-contradictory or the
  source/log evidence is recorded before the change.
- Remove inactive ancestry and diagnostic aliases only when they cannot alter
  the active graph.
- Show the old/new path before any change to input statistics, gradient
  geometry, attention competition, counterfactual or source bank.
- Do not use gates, quotas, entropy targets, artificial gradients or extra
  loss weight to hide an unresolved connection problem.

## Active identity

```text
capability:    object_intent_dynamics_323
schema:        30
topology:      3-2-3
intervals:     4-8 / 8-16 / 16-32 / 32-48
parameters:    measured and written per module at startup; never hard-coded
```

Schema 29 and older are not exact-resume sources for schema 30. Formal runs
start fresh unless the complete manifest, model, optimizer, scheduler and RNG
identity matches. Bottom-only migration is explicit and emits a report.

Schema30 keeps Schema27's bounded typed W normalization and Schema29's
detached self-conditioning/cache lifecycle, while closing the current
semantic boundaries. W's
ownership. W reads only a compact current object belief plus the normalized
physical action interval means and their adjacent deltas. Goal, S values and
coarse hidden tokens are absent from its API. P2 consumes an atomic
CandidateWorld whose action-condition identity must match the current cache.

Deployment and validation use one bounded outer correction: a complete ODE
proposal pass, one W rebuild from the decoded 24-row proposal, then a second
complete ODE pass from identical initial noise. The final action may differ
from the action that conditioned W, so interval/delta mismatch is logged as a
residual and is not labeled a fixed point. Training still builds W once.

S uses complementary K/type owners with one outer RMS contract; typed W keeps
learned chronology when the physical action is zero; camera support width is
metadata rather than a cross-camera vote; and validity is applied once at the
public W field boundary. Gripper trajectory supervision uses the exact deployed
absolute and qpos-anchored cumulative-delta codec branches. Event masks select
rows only and never reanchor deployment. Near-one capacity control remains FP32
through contraction. No event gate, capacity quota, new loss weight or extra
clipping stage is added.

## Runtime contract

- Observation/G/S, P1's N=49 detail read and the 512-row transition source
  build once per observation. Initial W builds once and only W is rebuilt once
  between the two deployment passes.
- In each pass the shared seed, dynamic P1, P2/P3, terminal contracts,
  ControlledTransition and bottom run at `[0,.2,.4,.6,.8]`; `t=1` supplies the
  retained motion head and cannot update action.
- Both passes share exact initial physical noise. Decoded gripper events are
  evaluation-only behavior of the second integrated physical action; there is
  no event classifier in runtime.
- Execution candidate/value charts are mandatory for every train/eval loss
  forward, even after the optional diagnostic-batch budget is exhausted; they
  stay disabled during both ordinary deployment ODE passes.
- Teacher builds once per training batch and zero times in deployment.
- P1 retains N=49 until each action/object query chooses a candidate; chunks
  are checkpointed rather than materializing the complete backward graph.
- Batch-eight production process peak must remain at or below 22 GiB.

## Run

Both launchers use the established server defaults:

```text
data:          /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
decoded cache: /data/senwang/data/cache_336
DINO cache:    /data/senwang/data/dinov2_cache_336
T5:            /data/senwang/checkpoint/grasp_pen_embed.pt
```

Smoke:

```bash
RUN_TAG=schema28_action_world_smoke_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/smoke_mainline.sh > "${RUN_TAG}.log" 2>&1 &
```

Formal batch-eight run:

```bash
RUN_TAG=schema28_action_world_b8_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/train_mainline.sh > "${RUN_TAG}.log" 2>&1 &
```

Each fresh output directory must be absent or empty. Override
`DATA_ROOT`, `CACHE_DIR`, `DINO_CACHE_DIR`, `T5_CONDITION_PATH`,
`MAINLINE_BATCH_SIZE` or `MAINLINE_NUM_WORKERS` only when the server layout
actually differs.

Audit the complete result rather than a best checkpoint:

```bash
uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema28_action_world_b8 \
  --recovery-baseline v120_long.log \
  --recovery-parent mainline_v120_contract_repair_b8.log \
  --tail 120 --require-recovery --format text
```

## Release gates

- full finite forward/backward and unique optimizer ownership;
- teacher isolation, two matched ODE passes and one W-rebuild frequency checks;
- physical-action/goal invariance, stale-world rejection and final-residual checks;
- typed axis, object permutation, P2 bounds and neutral consequence checks;
- fresh CUDA BF16 smoke and batch-eight memory measurement;
- complete eight-epoch comparison against V120, including action/native,
  first/tail, horizon bands, arm/gripper, event/motion, G/S/W/P and gradients;
- no early divergence or later rebound hidden by aggregate/best RMSE.
