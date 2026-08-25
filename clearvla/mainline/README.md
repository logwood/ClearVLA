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
  -> one typed-prebinding physical K+null grounder against current DINO
     (one public scene content value + K object-owned content innovations)
  -> S reads the public scene once, then K innovations, and forms observable
     interval innovations + per-type [interval,K,type] relevance
  -> typed clean action + causal W1-near/W2-far four-interval object dynamics
  -> P1 one cached protected-detail read over all progressive candidates
  -> per-ODE V120 P1 policy-query residual (static fact stays protected)
  -> P2 action-conditioned semantic-K / geometry-KxC spatial selection while
     preserving interval and type identity
  -> geometry supplies a bounded same-interval semantic-K address correction
  -> S conditions selected W keys; it owns no spatial/value/support/time vote
  -> P3 type-local physical four-interval terminal without learned null;
     W common and interval innovation are each read exactly once
  -> P3 protected dynamic-P1 precision + optional static precision,
     typed effect/temporal and state-change lanes
  -> shared V120 action/context canvas seed
  -> true P1/P2 terminal layer contracts
  -> exact completed-G3 anchor source + per-ODE noisy-action controlled transition
  -> V120 Evidence MMDiT / execution value / capacity bottom
```

Training adds one no-grad future-teacher branch. Future evidence has a separate
typed input and cannot enter the online/deployment API.

The 24-row executed-history future proposal is an auxiliary supervised
prediction in the recovered V120 object path. Its separate 4-recent +
3-summary history encoding remains the observable executed-history condition
used by the shared V120 canvas seed; it is not a P1 or controlled-action alias.
The object policy masks proposal from G/P/bottom, so the V120-compatible
proposal dropout RNG draw is retained but no deployment proposal ablation is
reported.

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
schema:        39
topology:      3-2-3
intervals:     4-8 / 8-16 / 16-32 / 32-48
parameters:    measured and written per module at startup; never hard-coded
```

Schema39 counts are measured rather than copied from an ancestral schema. The
launcher prints the total/trainable count and compact G/S/W/P/bottom summary,
then writes the complete per-module inventory into run context.

Schema38 is not an exact-resume, optimizer-resume or bottom-migration source for
Schema39 because the observation numerical boundary, P2/P3 terminal,
transition operand and bottom ingress ownership changed. Formal Schema39
comparisons start fresh in an absent or empty output directory.

The migration check covers the complete bottom key set, shapes, dtypes and
finite/non-complex tensor values before it mutates the live model; matching an
ABI label alone is not sufficient.

## Runtime contract

- Observation/G/S/W, P1's N=49 detail read and the 512-row transition source
  build once per observation.
- The shared action/context seed, compact P1 policy block, P2/P3, two terminal
  layer contracts, the action-conditioned transition and bottom run on every
  ODE step.
- Five action updates use times `[0,.2,.4,.6,.8]`. One additional full dynamic
  forward at `t=1` supplies event/motion heads and cannot update the action.
- Execution candidate/value charts are mandatory for every train/eval loss
  forward, even after the optional diagnostic-batch budget is exhausted; they
  stay disabled during ordinary five-step deployment sampling.
- Teacher builds once per training batch and zero times in deployment.
- P1 retains N=49 until each action/object query chooses a candidate; chunks
  are checkpointed rather than materializing the complete backward graph.
- Raw preclip diagnostics report the logging-window mean, max and current
  batch separately. A finite norm above the audit threshold emits a read-only
  parameter-attributed `gradient_spike` event before clipping.
- An epoch tail shorter than `log_every` is emitted as an explicit
  `window_boundary=epoch_tail` row, so its maximum gradient and owning step are
  retained.
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
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema39_action_closure_smoke \
nohup bash scripts/smoke_mainline.sh > schema39_action_closure_smoke.log 2>&1 &
```

Formal batch-eight run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema39_action_closure_b8 \
nohup bash scripts/train_mainline.sh > schema39_action_closure_b8.log 2>&1 &
```

Each fresh output directory must be absent or empty. Override
`DATA_ROOT`, `CACHE_DIR`, `DINO_CACHE_DIR`, `T5_CONDITION_PATH`,
`MAINLINE_BATCH_SIZE` or `MAINLINE_NUM_WORKERS` only when the server layout
actually differs.

Audit the complete result rather than a best checkpoint:

```bash
uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema39_action_closure_b8 \
  --recovery-baseline v120_long.log \
  --recovery-parent schema38_action_consumption_b8.log \
  --tail 120 --require-recovery --format text
```

## Release gates

Schema39 source implementation and contract tests do not imply training
recovery. The following gates still require a fresh run:

- full finite forward/backward and unique optimizer ownership;
- teacher isolation and five-step deployment call-frequency checks;
- typed axis, object permutation, P2 bounds and neutral consequence checks;
- fresh CUDA BF16 smoke and batch-eight memory measurement;
- complete eight-epoch comparison against V120, including action/native,
  first/tail, horizon bands, arm/gripper, event/motion, G/S/W/P and gradients;
- no early divergence or later rebound hidden by aggregate/best RMSE.
