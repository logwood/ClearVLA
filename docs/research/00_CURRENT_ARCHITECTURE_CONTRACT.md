# Current ClearVLA Architecture Contract

Updated: 2026-08-20

This is the compact source of truth for the active independent mainline.
Experiment labels never select model semantics. Historical evidence lives in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`; only still-open work belongs in
`CURRENT_MAINLINE_ISSUES.md`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        26
behavior reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
source reference:       .audit/v120_exact_source_0b92d359/
release status:         source-complete; local tests pass; awaiting CUDA smoke and behavior gates
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

> **Schema26 source is implemented, not yet behavior-released.** It closes the
> source-proven G3-anchor transmission defect and the S typed double-ingress,
> but a fresh CUDA smoke and complete eight-epoch comparison are still required
> before claiming action improvement. W interval under-resolution, early action
> lag and late gripper rebound remain behavior gates in
> [`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md).

> **Near-term scope lock:** do not make another source change before the fresh
> Schema26 smoke and behavior gates. If those gates fail after the exact G3
> boundary is confirmed, attribution starts at the S→W interval boundary; G,
> P1/P2/P3, transition core and bottom remain locked unless a new independent
> source defect is first recorded. Adjacency is not authorization to expand
> the repair scope.

The active graph lives in `clearvla/mainline/`. It does not dispatch through
the V39 trainer/runtime/trunk or a V-numbered capability branch. The manifest
owns serialized graph identity; typed interfaces and executable checks own
shape, dtype, provenance and zero semantics. Do not add a version-wide
`_validate_vXXX_*` contract.

Schema26 retains the controlled Schema24/V120 fidelity recovery, including its
flow-time, endpoint-head, Teacher algebra, support/selector split, non-finite
sentinel and the following four source-audited boundaries:

- literal G block -> progressive updater alternation at stages 1, 2 and 3;
- V120 P1 factorization with 24 factual queries, N=49 and a 3x3 microgrid;
- one dense global-object reconstruction objective;
- object-level online future geometry, with camera identity retained only
  where it is physically observed.

It also retains V120 AdamW decay and decoder-local then global clipping.
Schema26 keeps Schema25's S ownership boundary and makes four source-proven
repairs:

- S scores the public common carrier and zero-mean interval innovation
  separately, then combines them inside the selector's remaining bounded score
  range; the differential branch uses the zero-preserving 0.25 variance floor,
  so identical intervals remain zero and tiny residuals cannot be normalized
  into confident differences;
- semantic, appearance and geometry each compare only with a fixed zero null;
- full `[interval,K,type]` typed values enter W exactly once through
  `WorldIntentDock`; `ActionIntentDock` and CoarseAction cannot duplicate them;
- P1 and transition share the exact completed G3 rollout; transition no longer
  reconstructs four pseudo anchors from the public chart plus new identities;
- the unconsumed `future_address` export/online grid-sampling path and the
  bit-exact no-op proposal ablation are removed. The proposal dropout RNG draw
  remains solely for V120 generator-cadence compatibility.

No block, external loss weight, gain, quota, hard gate, entropy target,
capacity or P1 learned null was added. Schema25 and older checkpoints cannot
exact-resume Schema26.

The post-epoch-1 source/log closure audit then corrected four remaining
fidelity defects without changing the G/S/W/P or bottom topology:

- active V120 pre-G/address/future-query parameters are trainable again; only
  the unconsumed object-intent G3 generic route query remains frozen;
- Teacher transport/covariance now form displacement moments inside each
  camera before object-level reduction, so a camera-mass change cannot invent
  motion for a static object;
- the global-K binder no longer adds the public chart equally to every private
  candidate key;
- validation diagnostics are spread across the full loader; sampling and exact
  V120 execution ablations have separate coverage. The proposal path is absent
  from the V120 object policy and is no longer presented as a causal ablation.
  Primary deployment noise is restored to V120's deterministic per-batch
  stream (`37237 + one-based batch index`), and every ablation reuses that
  exact physical noise.

These corrections and the Schema26 transition/S boundary change alter the source and
state-dict fingerprints. Use a fresh output directory.

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
    -> StatelessObjectIntentOrganizer S
       protected goal/history/public-object values
       supervised public interval carrier [B,4,H]
       common/differential per-type fixed-zero relevance [B,4,K,3,*]
       consumer-specific policy interval context
       24 temporal queries and state-change evidence

ObjectFactSet public content/transport + S-owned WorldIntentDock (typed once)
    + causal clean CoarseActionIntent from public ActionIntentDock
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

exact completed G3 rollout shared with P1
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
   physical K=4 plus null competition. The public chart remains outside that
   private candidate competition and is never a second object value. Typed
   reads can reweight only inside physical K support.
6. Learned flow is a continuous source-relative prior, never a nonzero quota.
   Flow warp/cycle/smoothness/uncertainty/refinement keep explicit units.
7. S reads full T5, observable state/action history and typed G facts. It does
   not read frame progress, phase labels, noisy action or future Teacher. Its
   public carrier is supervised separately from optional typed relevance.
   Semantic, appearance and geometry retain real K/type axes, each owns a
   fixed-zero null comparison. Full K/type values enter W only through
   `WorldIntentDock`; CoarseAction is public-only. A named reduced policy
   context may enter the existing factual/policy consumers.
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
13. Transition static/dynamic frequency is model semantics: its current static
    source is the exact completed G3 rollout shared with P1 and builds once;
    real-versus-neutral coefficients read current noisy action at every dynamic
    forward. No public-chart reconstruction or second anchor identity is legal.
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

StatelessIntentBundle (serialized compatibility name: ObjectIntentState)
  protected goal/history/public-object tokens
  public / policy interval carriers                      [B,4,H]
  typed relevance mass / value                           [B,4,K,3,1|R]
    typed policy components                                [B,4,3,H]
  temporal queries / state-change evidence               [B,24,H] / [B,H]

Consumer views
  ActionIntentDock / WorldIntentDock / FactualIntentDock / PolicyIntentDock

FutureObjectDynamics
  current reference                                      [B,K,D]
  successor / semantic delta                             [B,4,K,D]
  transport / covariance                                 [B,4,K,2|3]
  visibility / persistence / uncertainty                 [B,4,K,1]
  future selector validity                               [B,4,K,1]

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
| W | public ObjectFactSet content/transport, S-owned typed relevance, one clean coarse action intent | raw semantic/appearance/geometry reread, target/noisy action, proposal, Teacher, free W residual |
| P1 | completed progressive chart, S, clean action bases | global-K value, W, proposal, Teacher, second visual read |
| P2 | completed P1 fact, supervised W field, S, noisy-action query | RGB/DINO reopen, camera-expanded W, free W hidden |
| P3 | P1 fact, consequence, S, noisy-action query | Teacher, RGB/DINO, proposal, free W carrier |
| transition source | exact completed G3 rollout | W target, proposal, noisy action, Teacher |
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
  raw posterior moments formed from same-camera
  `future_coordinate-current_camera_coordinate` displacements.
  Reliability/entropy do not shrink targets or masks.
- `current_loss_support [B,K,C,1]` owns future losses and the recognizer after
  detached camera reduction. `future_selector_validity [B,4,K,1]` belongs
  only to online P2 routing.
- Action, future, flow geometry, intent scaffold, history proposal and
  execution-value external weights are unchanged from the recovery reference.
- The whole-segment recognizer supervises only S's public interval carrier.
  Typed relevance is trained through the existing coarse-action, future W and
  final action paths; no public future target, entropy or usage loss directly
  trains its selector.
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
- The Schema26 formal configuration measures `169,979,847` total and
  `153,585,526` trainable parameters. Relative to Schema25, the exact `-2,048`
  delta is the removed transition-only four-row pseudo-anchor identity; no
  bottom, P1, G, S or W parameter was removed. Relative to the completed
  Schema24 graph, the exact `-12,733,181` trainable delta is fully accounted
  for: S removes
  three duplicate `_CrossRead`s plus one shared learned-null router and adds
  three route-width relevance projections plus three temperatures
  (`-6,308,093`); CoarseAction removes three duplicate `_CrossRead`s and one
  learned-null router (`-6,357,248`); W removes one learned-null router
  (`-65,792`). Bottom and the exact P1 reader are unchanged.
- Active manifest identity:

  ```text
  schema:       26
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_bank
  top:          v120_progressive_g123_dense_grounder_exact_p1_s_common_differential_k_typed_single_w_ingress_four_interval_w_five_lane_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_exact_g3_anchor_transition_evidence_mmdit_dense512_execution
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_v120_decay_local_global_clip
  runtime:      cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated_active_ablations_only
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
permutations, per-type S perturbation locality, fixed-zero typed values,
typed-owner relabeling equivariance, public-target gradient isolation,
absence of CoarseAction/W raw-typed rereads,
same-camera Teacher geometry, object geometry, neutral effect, P2 bounds, endpoint lifecycle,
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
OUT_DIR=runs/schema26_g3_s_boundary_smoke \
nohup bash scripts/smoke_mainline.sh > schema26_g3_s_boundary_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema26_g3_s_boundary_b8 \
nohup bash scripts/train_mainline.sh > schema26_g3_s_boundary_b8.log 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema26_g3_s_boundary_b8 \
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
