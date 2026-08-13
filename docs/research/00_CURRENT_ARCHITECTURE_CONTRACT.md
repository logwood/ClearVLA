# Current ClearVLA Architecture Contract

Updated: 2026-08-13

This is the compact source of truth for the active independent mainline.
Experiment labels never select model semantics. Historical evidence lives in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`; only still-open work belongs in
`CURRENT_MAINLINE_ISSUES.md`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        24
behavior reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
source reference:       .audit/v120_exact_source_0b92d359/
release status:         source-complete; awaiting CUDA smoke and behavior gates
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0, two adjacent learned flows
formal language:        precomputed 4096-wide T5 .pt required
bottom:                 V120 seed/transition/CVAE/workspace/Evidence MMDiT/execution
long launcher:          scripts/train_mainline.sh (batch 8, workers 4)
smoke launcher:         scripts/smoke_mainline.sh (batch 1, workers 0)
resolved config:        configs/mainline/object_intent_dynamics_323.json
```

The active graph lives in `clearvla/mainline/`. It does not dispatch through
the V39 trainer/runtime/trunk or a V-numbered capability branch. The manifest
owns serialized graph identity; typed interfaces and executable checks own
shape, dtype, provenance and zero semantics. Do not add a version-wide
`_validate_vXXX_*` contract.

Schema24 is a controlled V120 fidelity recovery. It retains the proven
Schema23 flow-time, endpoint-head, Teacher algebra, support/selector split and
non-finite sentinel, then restores the four source-audited boundaries:

- literal G block -> progressive updater alternation at stages 1, 2 and 3;
- V120 P1 factorization with 24 factual queries, N=49 and a 3x3 microgrid;
- one dense global-object reconstruction objective;
- object-level online future geometry, with camera identity retained only
  where it is physically observed.

It also restores V120 AdamW decay and decoder-local then global clipping. No
new block, loss weight, gain, quota, hard gate, entropy target, capacity or P1
learned null was added. Schema23 checkpoints cannot exact-resume Schema24.

## Active graph in execution order

```text
current RGB/DINO at -8,-4,0
    -> V120 raw/DINO compiler and two learned adjacent flows
    -> masked current evidence and full SoftAddressLatticeBank

current state + V120 visual rollout + one shared sampled role table
    -> observation-only grounding canvas
       (task/language/history/proposal/noisy-action slices are structurally empty)
    -> G1 DiT -> progressive update stage 1: coarse posterior/center/variance
    -> G2 DiT -> progressive update stage 2: rematerialize N=49 fine candidates
    -> G3 DiT -> progressive update stage 3: bounded owner correction
    -> completed camera x 8x8 x local-M GroundedFactSet
    -> competitive global K+null DenseObjectGrounder
    -> ObjectFactSet with reversible K <-> chart correspondence

T5 + observable state/executed-action history + ObjectFactSet
    -> cumulative StatelessObjectIntentOrganizer S
       protected goal/history/object values
       four interval queries, 24 temporal queries, state-change evidence

ObjectFactSet + S + causal clean CoarseActionIntent
    -> W1: 4-8 and 8-16
    -> W2: 16-32 and 32-48, causally reading W1
    -> one supervised object-level FutureObjectDynamics field

completed progressive chart + S + four clean action bases
    -> exact V120 LateRawDetailPolicyReader
       24 factual queries
       semantic/appearance/geometry/coverage glimpses
       complete N=49 posterior
       real 3x3 RGB/detail/coordinate microgrid
    -> organize after factual reading into [B,24,4,H]
    -> cached FactualPrecisionDock(protected_detail)

current noisy action + flow time + cached protected detail
    -> V120 dynamic P1 policy block at each dynamic forward
    -> completed live P1 fact

completed P1 fact + FutureObjectDynamics + S + noisy-action query
    -> P2 bounded content/intent/object-coordinate effect read
    -> zero-preserving protected consequence
    -> P3 factual/precision/effect/temporal/state-change lanes

one shared V120 action/context seed
    -> noisy-action query shared by P2/P3/transition/bottom
    -> current state, causal state history, compressed executed history

G3 chart
    -> static 512-row ControlledTransitionSource, once per observation
noisy action + V120 learned neutral + plan/history
    -> dynamic 512-row ControlledTransitionState, every dynamic forward

protected consequence + five P3 lanes + transition + shared seed
    -> V120 P1/P2 layer contracts
    -> CVAE/workspace/EvidenceViewAdapter
    -> three Evidence MMDiT blocks
    -> ordered low-rank contraction and execution-value controller
    -> 18-D physical velocity plus event/motion heads
```

The history-action proposal remains a supervised auxiliary prediction. Its
future proposal tokens do not enter G/S/W/P, transition or bottom. The
separately compressed executed-action history remains an observable condition
in the shared seed. Generic trajectory/workspace ingress is algebraic neutral;
protected consequence is written once, while the five P3 lanes are optional
typed evidence.

Training-only graph:

```text
future DINO supports
    -> FP32/no-grad Teacher association, once per training batch
    -> four object-level FutureObjectDynamics targets
future action/state + current_loss_support + teacher targets
    -> whole-segment recognizer and auxiliary losses only
```

Future evidence is absent from every online/deployment API. Replacing future
supports may change targets and losses, never deployment action.

## Non-negotiable invariants

1. V120 `long` is the default behavior reference. Change a mature mechanism
   only for a recorded source defect or after documenting input-distribution,
   gradient-geometry and rollback consequences.
2. Camera, space, local-M, global-K, progressive N=49, interval, horizon,
   action-basis and type axes remain real until a named consumer. A reduced
   axis may not be recreated with `expand` and renamed as original evidence.
3. G is current-only. It cannot read language, history proposal, noisy action
   or Teacher. G1/G2/G3 each cross a real progressive updater boundary; G2
   rematerializes N=49 exactly once and fresh G3 inherits its parent owner
   posterior when the bounded residual is zero.
4. The same sampled V120 role table seeds static G and every cached dynamic
   action call for that observation. G uses clean endpoint `t_v120=0`; ODE
   time cannot leak into cached facts.
5. Local M hypotheses are not persistent objects. The dense grounder owns one
   physical K=4 plus null competition. Public chart is address key only, never
   a second object value. Typed reads can reweight only inside physical K
   support.
6. Learned flow is a continuous source-relative prior, never a nonzero quota.
   Flow warp/cycle/smoothness/uncertainty/refinement keep explicit units.
7. S reads full T5, observable state/action history and typed G facts. It does
   not read frame progress, phase labels, noisy action or future Teacher.
8. W1 owns the two near intervals and W2 the two far intervals. The only W
   value below W is directly supervised `FutureObjectDynamics`; no public or
   private free W carrier crosses into P.
9. P1 owns 24 factual queries before action-basis organization, four factual
   glimpse types, the complete N=49 posterior and a real 3x3 microgrid.
   Global-K is not a P1 axis. `FactualPrecisionDock` is a parameter-free
   boundary containing only the already-computed protected detail; it is not a
   replacement reader or an extra bottleneck.
10. P2 geometry is object-level:

    ```text
    transport_mean/covariance       [B,4,K,*]
    object_coordinates              [B,K,2]
    future_selector_validity        [B,4,K,1]
    ```

    Teacher may associate per camera internally, but W cannot predict once and
    duplicate a fake camera axis. P2 cannot average camera squared distances
    into an implicit variance penalty.
11. Neutral effect is algebraically neutral:

    ```text
    effect = 0
    interaction = 0
    protected_consequence = completed P1 fact
    ```

12. P3 owns five V120 lanes: factual, precision, effect, temporal and
    state-change. It cannot reopen vision or consume a free W carrier.
13. Transition static/dynamic frequency is model semantics: static G3 source
    builds once; real-versus-neutral coefficients read current noisy action at
    every dynamic forward.
14. Bottom source count/order/value semantics follow V120. Do not remove CVAE,
    workspace, P1/P2 contracts, Evidence MMDiT, capacity or execution to reduce
    memory or simplify the mainline.
15. Online boundaries use ordinary autograd. No artificial gradient, hard
    gate, entropy/mass quota, scalar progress loss, forced diversity or forced
    nonzero flow is legal.
16. Formal training fails without the configured T5 file. Only explicit
    null-goal smoke may omit it.
17. Fresh runs require an empty output directory. Exact resume verifies
    manifest, source/data/language, model/optimizer/scheduler and RNG. Older
    schemas are rejected; explicit compatible bottom-only migration is the
    only migration path.

## Typed boundaries

```text
GroundingObservationBank
  current visual/value memory and SoftAddressLatticeBank

ProgressiveGroundingAddressState
  G1 coarse state
  G2 dynamic fine candidates                            [...,M,49,*]
  G3 completed GroundedFactSet                          [B,C,8,8,M,*]

ObjectFactSet
  content / semantic / appearance / geometry            [B,K,*]
  typed assignments                                     [B,K,C,8,8,M]
  observed camera coordinates/support/validity           [B,K,C,*]

ObjectIntentState
  protected goal/history/object tokens
  interval / temporal queries                            [B,4,H] / [B,24,H]
  state-change evidence                                  [B,H]

FutureObjectDynamics
  current reference                                      [B,K,D]
  successor / semantic delta                             [B,4,K,D]
  transport / covariance                                 [B,4,K,2|3]
  visibility / persistence / uncertainty                 [B,4,K,1]
  future selector validity                               [B,4,K,1]
  future address (diagnostic only)                       [B,4,K,C,8,8]

ObjectTopTrainingTargets
  current loss support                                   [B,K,C,1]

FactualPrecisionDock
  protected detail                                       [B,24,4,H]

ObjectPolicyPlanDeltaBank
  protected base + factual/precision/effect/temporal/state-change [B,24,4,H]

V120SeedContext
  state / state history / compressed executed history    [B,1|3|7,H]

ControlledTransitionSource / State
  static selector / dynamic selector and value            [B,512,H]
```

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| G | current DINO/raw history, coordinates, learned flow, current state | T5, action history, proposal, noisy action, Teacher |
| global grounder | completed G3 chart and typed local candidates | S, W, noisy action, Teacher |
| S | T5, state/executed history, typed ObjectFactSet | frame progress, phase label, noisy action, Teacher |
| W | ObjectFactSet, S, one clean coarse action intent | target/noisy action, proposal, Teacher, free W residual |
| P1 | completed progressive chart, S, clean action bases | global-K value, W, proposal, Teacher, second visual read |
| P2 | completed P1 fact, supervised W field, S, noisy-action query | RGB/DINO reopen, camera-expanded W, free W hidden |
| P3 | P1 fact, consequence, S, noisy-action query | Teacher, RGB/DINO, proposal, free W carrier |
| transition source | G3 public chart | W target, proposal, noisy action, Teacher |
| transition dynamic | source, shared V120 seed, plan | target action, Teacher, future proposal |
| bottom | consequence, five P3 lanes, transition, seed, layer contracts | RGB/DINO, Teacher, duplicate W/P base |

## Loss and optimizer ownership

- Physical V120 action flow matching remains dominant.
- The global grounder owns exactly one dense-mixture reconstruction MSE; the
  existing intent-structure ledger applies its fixed 0.25 internal coefficient.
  No prototype, masked-completion or typed-consistency head remains.
- Teacher successor is the uniform interval mean of
  `matched + null_probability * current_reference`; semantic delta is exactly
  successor minus current reference. Transport/covariance are uniform means of
  raw posterior moments. Reliability/entropy do not shrink targets or masks.
- `current_loss_support [B,K,C,1]` owns future losses and the recognizer after
  detached camera reduction. `future_selector_validity [B,4,K,1]` belongs
  only to online P2 routing.
- Action, future, flow geometry, intent scaffold, history proposal and
  execution-value external weights are unchanged from the recovery reference.
- Every trainable parameter has exactly one optimizer owner. Ordinary bias,
  LayerNorm, top/controller/decoder parameters use AdamW decay 0.01. Only
  explicitly named scale-invariant contraction basis/depth coordinates are
  no-decay.
- Gradient lifecycle is:

  ```text
  finite check -> gradient_raw
  -> bottom.decoder local clip 1.0 -> gradient_postlocal
  -> global clip 1.0 -> gradient_postglobal
  ```

  A non-finite batch records the first named parameter, role, optimizer group,
  dtype/shape and NaN/Inf statistics before any optimizer, scheduler or step
  update.

## Runtime, identity and inventory

- Observation/G/S/W, exact static P1 and transition source build once per
  observation.
- Dynamic P1/P2/P3, transition, layer contracts and bottom run at action-update
  times `[0,.2,.4,.6,.8]`, then once at `1.0` for event/motion heads only.
  The endpoint call cannot change the integrated action.
- Teacher builds once per training batch and zero times in deployment.
- P1 N=49 queries use the V120 query budget/checkpoint configuration.
  Chunked and unchunked outputs and parameter gradients must be equivalent.
- Startup writes a per-module parameter inventory. Counts are measured, never
  hard-coded into the contract; any difference from V120 must name the removed
  and restored owners.
- Active manifest identity:

  ```text
  schema:       24
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_bank
  top:          v120_progressive_g123_dense_grounder_exact_p1_object_geometry_four_interval_w_five_lane_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_evidence_mmdit_dense512_execution
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_v120_decay_local_global_clip
  runtime:      cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated
  ```

Storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache/checkpoint roots moved.

## Verification and run

Local tests cover full forward/backward, G1/G2/G3 ordering and N=49
rematerialization, forbidden G conditions, exact P1 axes/microgrid,
chunked/unchunked P1 output and gradients, Teacher isolation, object/camera
permutations, object geometry, neutral effect, P2 bounds, endpoint lifecycle,
optimizer ownership, three-stage gradient logging and checkpoint rejection.
CPU BF16 validates dtype boundaries, not CUDA memory.

Production acceptance still requires:

- fresh BF16 smoke and five-step deployment;
- batch-eight process peak no greater than 22 GiB;
- aligned batch-2200 early recovery gate against V120;
- all eight epochs and final/mean action, native, first/tail, horizon,
  arm/gripper, event/motion, G/S/W/P and gradient comparisons;
- no late rebound hidden by a best checkpoint.

Use new empty output directories:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema24_v120_fidelity_smoke \
nohup bash scripts/smoke_mainline.sh > schema24_v120_fidelity_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema24_v120_fidelity_b8 \
nohup bash scripts/train_mainline.sh > schema24_v120_fidelity_b8.log 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema24_v120_fidelity_b8 \
  --recovery-baseline v120_long.log \
  --recovery-parent mainline_v120_contract_repair_b8.log \
  --tail 120 --require-recovery --format text
```

## Authoritative source map

```text
identity/config/interfaces:
  clearvla/mainline/manifest.py
  clearvla/mainline/config.py
  clearvla/mainline/interfaces.py
observation/G:
  clearvla/mainline/model/restored_observation.py
  clearvla/mainline/model/observation_contract.py
  clearvla/mainline/model/action_contract.py
S/W/P:
  clearvla/mainline/model/top.py
  clearvla/mainline/model/grounding.py
  clearvla/mainline/model/intent.py
  clearvla/mainline/model/dynamics.py
  clearvla/mainline/model/teacher.py
  clearvla/mainline/model/v120_p1.py
  clearvla/mainline/model/compiler.py
bottom/runtime:
  clearvla/mainline/model/restored_bottom.py
  clearvla/mainline/model/transition.py
  clearvla/mainline/training/
  clearvla/mainline/runtime/
```
