# Current ClearVLA architecture contract

Updated: 2026-09-02

This is the compact source of truth for the active independent mainline. Read
it before changing the V96+ top representation, Flow-DINO/JEPA, role hierarchy,
language/history conditioning, long-horizon handling or the top-to-bottom
evidence path. Live process state belongs in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md);
open behavior questions and execution order belong in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md) and
[`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md).

The full pre-compaction contract remains recoverable at Git commit `f60bd80`.
Historical experiment names never select current semantics.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        30
manifest digest:        1323dcff095cbddb8da02c0e263c3e9865fbae39add9af4e539d38e9745f9c46
Linux source digest:    0d0957a75ab22e37f552ccf9a4505049876af5785837cb9787edde181b04c1c2
Schema30 source commit: 3fef2fc0dce297f600c813307c998f587cca1ca3
formal-run checkout:    f60bd808becabd882b10ad7b07e74242fe49a881
branch:                 codex/schema29-mainline (historical branch name only)
behavior reference:     Schema28, commit 097330a894d948d66c419f8af07325a5b0ff712e
recovery reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global objects:         K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0; two adjacent learned flows
training:               fresh, single-stage end-to-end, eight epochs
bottom:                 V120 seed/transition/CVAE/workspace/Evidence MMDiT/execution
long launcher:          scripts/train_mainline.sh
RDT-8 launcher:         scripts/train_rdt_multitask.sh
smoke launchers:        scripts/smoke_mainline.sh / scripts/smoke_rdt_multitask.sh
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only)
config:                 configs/mainline/object_intent_dynamics_323.json
```

Release state: Schema30 source, local tests/static gates, real Pen B8 CUDA VJP,
Pen/RDT-8 smokes and both read-only checkpoint validations have passed. Fresh
Pen and RDT-8 formal runs are active. Schema29 and older checkpoints are not
Schema30 exact-resume or migration inputs.

## Authority order

When documents disagree, use this order:

1. active source plus the supplied run's serialized `run_context.json`;
2. this contract;
3. current issues and repair plan;
4. the two current detailed references: RDT adaptation and R1/R2 closure index;
5. archived replay/design/log documents and Git history.

An old filename, log banner, launcher comment or conversation statement cannot
override manifest, source, config, dataset and checkpoint identity.

## Active graph

### Observation and grounding

```text
RGB/DINO at -8,-4,0
  -> V120 raw/DINO compiler + two learned adjacent flows
  -> current-only G1/G2/G3 progressive grounding
     G2 rematerializes N=49 fine candidates exactly once
  -> camera x 8x8 x local-M hypotheses
  -> dense global K=4 + null grounder
  -> ObjectFactSet and reversible K <-> chart correspondence
```

G is current-only: it cannot read language, executed-history proposal, noisy
action or future Teacher evidence. Local-M rows are hypotheses, not persistent
objects. The dense grounder owns physical real/null mass; G3 may refine only
`P(K | real)`. Reconstruction uses detached current DINO over observed cells
and exports one K-specific content value shared by reconstruction, S, W and the
detached Teacher.

### Intent, physical action and world

```text
T5 + observed state/executed history + ObjectFactSet
  -> S public interval carrier + typed [interval,K,type] relevance
  -> typed-free CoarseAction physical proposal [B,4,7]
  -> PhysicalActionCondition
     absolute interval means + current-anchored adjacent deltas [B,4,14]
  -> W(ObjectWorldBelief, PhysicalActionCondition)
     W1 owns intervals 0/1; W2 reads W1 and owns intervals 2/3
  -> action-tagged CandidateWorld / FutureObjectDynamics
```

W cannot read goal, S values, coarse hidden tokens, Teacher or noisy ODE
action. Semantic successors remain `[B,4,K,D]`; transport/covariance and
camera support remain camera-resolved `[B,4,K,C,*]` until P2. W predicts no
visibility, status or validity authority.

### P1, P2, P3, transition and bottom

```text
completed progressive chart
  -> one cached V120 P1 high-resolution read
     24 factual queries, N=49, real 3x3 RGB/detail microgrid
  -> FactualPrecisionDock

noisy action + time + cached factual detail
  -> dynamic P1 policy residual
  -> P2 semantic-K and geometry-K*C selection inside each interval
  -> one no-null physical interval terminal per type
  -> semantic + geometry consequence

P2 consequence + S/action context -> optional P3 temporal innovation
observed state change + S/action context -> optional P3 state-change innovation

completed G3 rollout -> cached ControlledTransitionSource
noisy action + consequence + dynamic P1 residual -> dynamic transition
all protected/optional carriers -> V120 bottom -> physical velocity/motion
```

Protected consequence and raw dynamic P1 precision are no-null carriers.
Only temporal and state-change P3 lanes own zero-null choices, through separate
invocations of one shared reader. The Evidence MMDiT, continuous capacity and
execution-value machinery remain intact.

### Training-only Teacher and self-conditioning

Future DINO supports enter only the no-grad Teacher and auxiliary targets.
Teacher builds once per training batch and zero times in deployment.

Schema29 introduced, and Schema30 retains, one detached training-side endpoint
self-conditioning lifecycle:

```text
sample one noisy FlowMatchingState
  -> pass0 velocity under no-grad, AMP weight cache disabled, forked RNG
  -> decoded clean endpoint -> detached PhysicalActionCondition
  -> rebuild W only
  -> pass1 formal velocity from the same noisy field/time, AMP cache enabled
  -> compose action/future/auxiliary losses once from pass1
```

Pass0 has no loss and cannot own a backward edge. RNG restoration leaves the
global stochastic stream equal to one formal dynamic pass. Disabling the AMP
weight cache is local to parameterized no-grad scopes; formal computation keeps
the normal cache.

### Deployment and validation

Deployment performs exactly two complete five-update ODE passes from identical
initial physical noise:

```text
W(coarse) -> complete proposal ODE -> decoded 24-row proposal
          -> deterministic 24-to-4 PhysicalActionCondition -> rebuild W once
          -> complete refined ODE -> final action
```

This is one bounded correction, not a fixed point. The final action may differ
from the action that conditioned the rebuilt W; interval/delta mismatch is a
required residual metric. Recomputing `W(final)` without a later policy consumer
does not close the loop.

## Schema30 semantic delta

Schema30 changes source/config semantics without adding a block, parameter,
buffer, optimizer group, loss weight, RNG draw or deployment pass:

1. S sums complementary K and semantic/appearance/geometry owner axes, then
   applies the existing outer RMS contract once. It does not divide by an
   assumed active-owner count.
2. Typed W interval innovations read learned chronology together with the
   physical action condition; zero action no longer erases interval identity.
3. `camera_support` remains geometry width/uncertainty metadata. Only
   producer-owned `camera_validity` participates in cross-camera reduction.
4. Gripper trajectory supervision uses the exact deployed absolute and
   qpos-anchored cumulative-delta codec operands. Masks select rows only.
5. Continuous object/camera validity is applied once at the public W field
   boundary; private typed owners remain continuous and zero support still
   exports exact zero.
6. P3 optional `source_depths` is built through the public cardinality-checked
   compiler path.
7. The unconsumed `proposal_condition_dropout` field is removed.

These are ownership/semantic repairs, not numerical hardening. Do not add a
gain, quota, hard event gate, entropy target, extra clipping stage or objective
weight merely to make one logged magnitude look larger.

## Non-negotiable invariants

1. Camera, spatial, local-M, global-K, N=49, interval, horizon, basis and type
   axes remain real until a named consumer. A reduced axis cannot be recreated
   with `expand` and called original evidence.
2. Online evidence is ordinary autograd unless this contract names a no-grad
   Teacher/audit scope. Nonzero activation gradients do not substitute for
   parameter-owner VJP.
3. Learned flow is a continuous source-relative prior, never a forced-nonzero
   target or route quota.
4. S is the only intent owner. It cannot create W value/support or enter W as a
   second hidden path.
5. W owns the only future object field. ControlledTransition consumes policy
   transition evidence; it does not produce `world1` and has no extra W bridge.
6. Physical validity/support is producer-owned and applied once. Confidence,
   allocation share and support width are not interchangeable vote weights.
7. P1 retains its high-resolution N=49 and 3x3 read. It cannot be replaced by a
   K-object summary merely to save memory.
8. Semantic K and geometry K*C selection are independent and complementary.
   They do not compete in a type softmax and physical interval selection has no
   learned null.
9. Neutral P2 effect is algebraically neutral:
   `effect=0`, `interaction=0`, `protected_consequence=factual_base`.
10. Bottom V120 seed, terminal layer contracts, CVAE/workspace, Evidence MMDiT,
    capacity and execution remain present. Capacity is a continuous numerical
    contract, not a claim of hardware rank reduction.
11. Deployed gripper behavior comes from continuous physical value/delta
    branches. Decoded events are evaluation metrics, never a runtime gate.
12. Future observation/action/state evidence may affect detached Teacher or
    training targets only; replacing it cannot change deployment action.
13. One optimizer owns every trainable parameter exactly once. Decoder-local
    clipping precedes global clipping; finite post-clip values cannot hide a
    missing raw owner gradient.
14. Formal runs fail closed on missing language mappings, source/config/
    manifest mismatch, stale CandidateWorld identity and non-finite values.
15. Checkpoints, tensor caches, raw logs and full probe dumps never enter
    architecture-memory documents.

## Typed boundary summary

| Boundary | Required semantics |
|---|---|
| `ObjectFactSet` | K=4 physical objects plus explicit null; one exported content value; observable object/camera probability and log probability |
| `ActionIntentDock` | public S context only; no typed fact re-entry |
| `PhysicalActionCondition` | four absolute physical interval means plus current-anchored adjacent deltas, `[B,4,14]` |
| `ObjectWorldBelief` | compact current G belief; no goal/S/Teacher/noisy-action field |
| `CandidateWorld` | atomic action-condition identity plus one `FutureObjectDynamics` |
| `FutureObjectDynamics` | semantic successor/delta and camera-resolved transport/covariance; copied current validity only |
| `FactualPrecisionDock` | already-computed protected P1 detail; no new reader or compression |
| `CompletedP1PolicyState` | static factual base separate from dynamic noisy-action/time residual |
| `SelectedIntervalEvidence` | interval-retaining semantic/geometry values; no-null physical terminal |
| `ControlledTransitionSource` | exact completed G3 rollout built once per observation |

## Loss, gradient and optimizer ownership

The logged loss ledger is authoritative. Raw auxiliary magnitudes do not imply
optimization dominance; interpret `loss_contrib_*` and `loss_group_*` first.
The main groups are action, representation and execution. The retained `.03`
gripper-trajectory budget supervises continuous transition/persistence on the
deployed codec branches; it does not fund an event classifier.

Diagnostics and matched interventions are audit-only unless the source shows
an explicit positive objective weight. Every train window must keep the ledger
closed, raw owner gradients finite and each named optimizer role present.

Schema30 retains the Schema28 inventory:

```text
total parameters:        168,417,179
trainable parameters:    152,046,448
parameter tensors:       1,385
trainable/optimizer:      1,063
optimizer groups:        23
state-key names:          1,391
```

## Data outlets

### Pen core-behavior outlet

```text
raw HDF5:       /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
decoded cache:  /data/senwang/data/cache_336
DINO cache:     /data/senwang/data/dinov2_cache_336
T5 condition:   /data/senwang/checkpoint/grasp_pen_embed.pt
split:          63 train / 5 val / 5 test episodes
batch/workers:  8 / 4
normalizer v120 fingerprint: 32a3a4d7f21f
```

This outlet answers core closure: far horizon, gripper, S/W/P, refinement and
gradient health.

### RDT-8 external-interface outlet

```text
raw root:       /data/rdt-ft-data
model cameras:  high + right_wrist
model action:   right arm 7-D projection from native 14-D
T5 bank:        /data/senwang/data/rdt_ft_data/multitask_v1/t5_v1_1_xxl_32.pt
train/val/test: 54,648 / 6,711 / 6,990 windows
sampling:       eight-task balanced; one row per task in every B8 batch
validation:     64 rows per task, 512 rows total
```

Task identity is used for sampling, validation and logging only; it is not a
hidden model condition. This outlet validates the adapter and cross-task
ecology. It does not claim native three-camera, depth or bimanual 14-D model
consumption. Details live in
[`auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md`](auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md).

## Identity and checkpoint contract

- The manifest, resolved config, executable-source digest, dataset inventory,
  normalizers, language artifact, optimizer ownership and RNG state are
  serialized in `run_context.json`/checkpoint metadata.
- Branch and run-directory names are descriptive only. The current branch name
  still contains `schema29`; the manifest must report Schema30.
- Schema29 and earlier checkpoints are rejected for exact resume. Smoke
  checkpoints are gate artifacts, not formal initialization sources.
- `validate_mainline_checkpoint.sh` is read-only: optimizer, scheduler and RNG
  load are disabled and no checkpoint is written.
- Formal output directories must be new and empty. Checkpoint writes are atomic;
  do not overwrite an existing run to continue a different identity.

## Release evidence

All required Schema30 pre-training gates passed:

| Gate | Result |
|---|---|
| local regression/static | `223 passed, 2 CUDA-only skipped`; changed-file Ruff and compileall pass |
| checkpoint compatibility | fresh save/load round-trip passes; Schema29 exact resume rejected |
| real Pen B8 CUDA VJP | cache0/cache1 total parameter L2 `3.1326139 / 3.1326158`; velocity, gripper, motion and MMDiT owners retained |
| Pen B8 smoke | `schema30_pen_b8_smoke_20260902_112950`; exact ledger, finite backward, atomic checkpoints, 4.228 GiB peak estimate |
| RDT-8 smoke | `schema30_rdt8_smoke_20260902_113250`; exact ledger, 8/8 coverage, finite backward, 10.53 GiB peak estimate |
| Pen checkpoint validation | `schema30_pen_checkpoint_validation_20260902_113954`; `source_delta_files=0`, read-only lifecycle |
| RDT checkpoint validation | `schema30_rdt8_checkpoint_validation_20260902_114122`; `source_delta_files=0`, read-only lifecycle |

The VJP gate exists because the first Schema29 run exposed a CUDA BF16 AMP
weight-cache failure: pass0 no-grad casts severed formal parameter edges while
forward values and activation gradients stayed finite. Commit `d8a77a1` closed
that lifecycle defect. A finite total gradient or optimizer step never replaces
the real parameter-owner VJP gate.

These are release/interface results, not behavior results. Schema28 remains the
completed behavior anchor until Schema30 finishes its full curves.

## Run and audit

Canonical commands and the current remote environment are maintained in
[`clearvla/mainline/README.md`](../../clearvla/mainline/README.md) and the live
handoff. Formal runs use the XVLA Python environment on the server; non-
interactive SSH must make that environment visible in `PATH`.

Hard stops are: non-finite values, lineage/identity failure, an open loss
ledger, formal parameter-owner VJP disappearance, checkpoint ABI violation,
memory above the 22 GiB release boundary, or a persistent severe spike pattern.
One finite cold-start output-head crossing, early event F1, small geometry RMS
or capacity warmup does not independently stop a run.

## Authoritative source map

```text
identity/config/interfaces:
  clearvla/mainline/manifest.py
  clearvla/mainline/config.py
  clearvla/mainline/interfaces.py
observation/G:
  clearvla/mainline/model/restored_observation.py
  clearvla/mainline/model/observation_contract.py
  clearvla/mainline/model/grounding.py
S/W/P:
  clearvla/mainline/model/intent.py
  clearvla/mainline/model/dynamics.py
  clearvla/mainline/model/policy.py
  clearvla/mainline/model/v120_p1.py
  clearvla/mainline/model/compiler.py
Teacher/transition/bottom:
  clearvla/mainline/model/teacher.py
  clearvla/mainline/model/transition.py
  clearvla/mainline/model/restored_bottom.py
training/runtime:
  clearvla/mainline/training/
  clearvla/mainline/runtime/
  clearvla/mainline/train.py
```

Historical replay provenance is indexed by
[`auxiliary/R1_R2_CLOSURE_INDEX.md`](auxiliary/R1_R2_CLOSURE_INDEX.md). Open the
long replay archive only for ancestry, an old log or the reason behind a past
repair; never reconstruct the active graph from it.
