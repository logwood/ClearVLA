# Current ClearVLA Architecture Contract

Updated: 2026-08-21

This is the compact source of truth for the active independent mainline.
Experiment labels never select model semantics. Historical evidence lives in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`; only still-open work belongs in
`CURRENT_MAINLINE_ISSUES.md`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        29
behavior reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
source reference:       .audit/v120_exact_source_0b92d359/
release status:         local implementation verified; fresh CUDA smoke required before release
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

> **Schema29 is a focused P2 ownership repair over the integrated Schema28
> graph, not another broad redesign.** Schema28 was one integrated
> ownership/dataflow repair, not another additive mechanism. Complete
> Schema25/26/27 logs plus the source audit identified
> conditional-K, fixed-template S, typed W, per-type P2 and P3 source-semantic
> faults. Schema28 repairs only those recorded boundaries. The exact V120 P1
> reader, transition and complete V120 bottom remain locked.

The active graph lives in `clearvla/mainline/`. It does not dispatch through
the V39 trainer/runtime/trunk or a V-numbered capability branch. The manifest
owns serialized graph identity; typed interfaces and executable checks own
shape, dtype, provenance and zero semantics. Do not add a version-wide
`_validate_vXXX_*` contract.

Schema29 retains the controlled Schema24/V120 fidelity recovery and complete
Schema28 G/S/W/P ownership graph, including its
flow-time, endpoint-head, Teacher algebra, support/selector split, non-finite
sentinel and the following four source-audited boundaries:

- literal G block -> progressive updater alternation at stages 1, 2 and 3;
- V120 P1 factorization with 24 factual queries, N=49 and a 3x3 microgrid;
- one dense global-object reconstruction objective;
- object-level online future geometry, with camera identity retained only
  where it is physically observed.

It also retains V120 AdamW decay and the exact three-owner clipping lifecycle.
Schema28 applied the following source-audited changes as one serialized graph
identity:

- G3 corrects only the conditional K distribution and preserves G2's exact
  object-vs-null mass. The global grounder reconstructs the independent
  observed current-DINO chart as a protected public mean plus exported
  K-specific residuals; no private reconstruction decoder exists. The online
  typed boundary exports that public content once and gives S/W a separate K
  innovation axis. Absolute K content is retained only as the detached
  current-reference field used by Teacher/W identity bookkeeping; it is not
  an online S/W/P2 value source;
- semantic, appearance and geometry may only reweight candidates inside the
  physical K support. Camera support width, read-conditioned physical
  `camera_validity`, and pre-normalization assignment evidence are separate
  fields. Evidence mass owns camera reduction/audit; physical camera validity
  owns future and typed-S loss support;
- object existence is a detached prior only for optional online future
  candidacy. It never masks the protected current fact, Teacher/future losses,
  or a value tensor, so learned null cannot become a global erasure shortcut;
- S learned identities are query addresses only. Goal, common interval and
  differential interval values are zero-preserving observable innovations;
  zero T5 is an exact language null. State/action history is one typed
  time-union rather than fake paired rows, and full `[interval,K,type]` values
  enter both the clean coarse action and W boundaries;
- W semantic/appearance/geometry states actually traverse W1/W2 and are
  decoded directly by the same field heads used by direct S/W supervision.
  Final public W hidden cannot re-publicize or suppress those typed states;
- P2 owns independent semantic, geometry and status source posteriors/nulls.
  Its public interval key is the untyped condition innovation; typed values
  enter only through the matching per-type key and are not duplicated through
  the public key.
  Geometry can positively support a coordinate match, while disappearance is
  selected from current support instead of masking its own status value.
  The three selected values are complementary rather than mutually exclusive:
  their symmetric mean is a protected fusion base and a bias-free low-rank
  LayerScale residual may read only type contrasts. There is no outer type
  softmax/gate; all-null stays exact zero and identical typed values cannot be
  rewritten by the residual;
- the protected P1+effect consequence remains outside optional P3 routes.
  The inactive factual pseudo-null lane is removed. Precision is action times
  the cached high-resolution P1 reader innovation; effect is W-effect only;
  temporal requires S, protected consequence and action multiplicatively;
  state-change keeps its existing exact-zero source semantics;
- Teacher covariance is the full candidate-plus-identity-null mixture moment;
  active G boundaries and physically distinct candidate key/value magnitudes
  replace the stale/aliased console metrics.
- frame-progress diagnostics read the exact supervised S condition innovation,
  never the fixed learned interval-address carrier; progress remains audit-only.

No external loss weight, block count, quota, hard gate, entropy target,
capacity or P1 learned null is added. Schema28 and older checkpoints cannot
exact-resume Schema29.

Retained recovery boundaries include the exact completed-G3 rollout shared by
P1 and transition, removal of the unconsumed `future_address`/online
grid-sampling path, and retention of the proposal-dropout RNG draw solely for
V120 generator-cadence compatibility. These are ancestry facts, not alternate
active routes.

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
    -> typed pre-binding consensus over semantic/appearance/geometry
    -> one competitive global K+null DenseObjectGrounder
    -> ObjectFactSet with reversible K <-> chart correspondence
       one public_content [B,D] + K content_innovation [B,K,D]
       absolute K content remains only as detached Teacher/W current-reference
       bookkeeping, not as an online S/W/P2 value source

T5 + observable state/executed-action history + ObjectFactSet
    -> StatelessObjectIntentOrganizer S
       exact-null goal innovation + typed time-union history
       public scene content is read once; K reads contain only innovations
       learned interval identities used only as query addresses
       observable interval-condition innovation [B,4,H]
       common/differential per-type signed zero-null values [B,4,K,3,*]
       typed values enter clean CoarseAction and W without K/type reduction
       24 zero-preserving temporal innovations and state-change evidence

one ObjectFactSet public content + K innovations/transport
    + S-owned WorldIntentDock (typed once)
    + causal clean CoarseActionIntent from public+typed ActionIntentDock
    -> W1: 4-8 and 8-16, field-matched typed states cross the block
    -> W2: 16-32 and 32-48, typed states causally read W1
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
    -> P2 independent semantic/geometry/status bounded source reads
    -> zero-preserving protected consequence
    -> P3 precision/effect/temporal/state-change innovations

one shared V120 action/context seed
    -> noisy-action query shared by P2/P3/transition/bottom
    -> current state, causal state history, compressed executed history

exact completed G3 rollout shared with P1
    -> static 512-row ControlledTransitionSource, once per observation
noisy action + V120 learned neutral + plan/history
    -> dynamic 512-row ControlledTransitionState, every dynamic forward

protected consequence + four active P3 lanes + transition + shared seed
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
protected consequence is written once, while the four P3 lanes are optional
typed evidence.

Training-only graph:

```text
future DINO supports
    -> FP32/no-grad Teacher association, once per training batch
    -> four object-level FutureObjectDynamics targets
future state + current_loss_support + Teacher fields
    -> direct public-state and typed-boundary S supervision only
future action
    -> clean CoarseAction supervision only
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
   compatibility is evaluated before binding, but its consensus creates only
   one physical assignment; typed reads can refine only inside that support.
   The reconstruction target is the independent observed current-DINO chart.
   `ObjectFactSet` exports one public content value and K object innovations;
   repeating the public direction as K owned values is forbidden. Absolute K
   content is retained only for Teacher's current-reference identity.
6. Learned flow is a continuous source-relative prior, never a nonzero quota.
   Flow warp/cycle/smoothness/uncertainty/refinement keep explicit units.
7. S reads full T5, observable state/action history and typed G facts. It does
   not read frame progress, phase labels, noisy action or future Teacher. Its
   condition innovation predicts canonical future-state summaries; the exact
   typed S→W boundary predicts matching detached Teacher fields during
   training. Goal/interval learned tokens are query addresses, not values;
   zero T5 and zero observable interval evidence remain exact zero.
   The public scene is one S memory row and K content rows are public-free
   innovations. Semantic, appearance and geometry retain real K/type axes and signed
   zero-null semantics. Full K/type values enter W through `WorldIntentDock`
   and the clean-action compiler through `ActionIntentDock`; neither may
   reread raw G fields or collapse type/K before its named query.
8. W1 owns the two near intervals and W2 the two far intervals. S owns the one
   learned interval coordinate. W's semantic, appearance and geometry typed
   states cross both blocks without mixing their type axis and cannot
   substitute for one another before matching output fields. The only W value below W is directly supervised
   `FutureObjectDynamics`; no public or private free W carrier crosses into P.
   W may reconstruct an absolute object coordinate only as the explicit sum
   of the single public projection and each K innovation projection.
9. P1 owns 24 factual queries before action-basis organization, four factual
   glimpse types, the complete N=49 posterior and a real 3x3 microgrid.
   Global-K is not a P1 axis. `FactualPrecisionDock` is a parameter-free
   boundary containing only the already-computed protected detail; it is not a
   replacement reader or an extra bottleneck.
10. P2 geometry is object-level:

    ```text
    transport_mean/covariance       [B,4,K,*]
    object_coordinates              [B,K,2]
    current_selector_validity       [B,K,1]
    future_selector_validity        [B,4,K,1]
    ```

    Teacher may associate per camera internally, but W cannot predict once and
    duplicate a fake camera axis. Semantic, geometry and status have separate
    source posteriors/nulls; status uses current support so disappearance
    cannot mask itself. They do not enter a second competitive selector:
    a protected mean plus a near-zero low-rank type-contrast residual performs
    the only final fusion. P2 cannot average camera squared distances into an
    implicit variance penalty.
11. Neutral effect is algebraically neutral:

    ```text
    effect = 0
    interaction = 0
    protected_consequence = completed P1 fact
    ```

12. P3 exposes four active optional rows; the old all-zero factual pseudo-null
    row is absent because factual consequence is already protected. Precision
    owns action×cached-P1-detail, effect owns W effect, temporal requires
    S×protected-consequence×action, and state-change owns its observed
    zero-centred source. It cannot reopen vision, duplicate the protected base
    or consume a free W carrier.
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
  public content                                         [B,D]
  absolute content (Teacher reference only)              [B,K,D]
  content innovation / semantic / appearance / geometry  [B,K,*]
  typed assignments                                     [B,K,C,8,8,M]
  observed camera coordinates/support/validity/evidence  [B,K,C,*]

StatelessIntentBundle (serialized compatibility name: ObjectIntentState)
  protected goal innovation / typed time-union history
  one public scene token / K content innovations          [B,1,H] / [B,K,H]
  public address carrier / condition+policy innovations  [B,4,H]
  typed relevance mass / value                           [B,4,K,3,1|R]
  typed policy components                                [B,4,3,H]
  temporal innovations / state-change evidence           [B,24,H] / [B,H]

Consumer views
  ActionIntentDock / WorldIntentDock / FactualIntentDock / PolicyIntentDock

FutureObjectDynamics
  current reference                                      [B,K,D]
  successor / semantic delta                             [B,4,K,D]
  transport / covariance                                 [B,4,K,2|3]
  visibility / persistence / uncertainty                 [B,4,K,1]
  current selector validity                              [B,K,1]
  future selector validity                               [B,4,K,1]

ObjectTopTrainingTargets
  current loss support                                   [B,K,C,1]
  public future-state target                             [B,4,S]
  typed semantic/status/transport targets                [B,4,K,*]

FactualPrecisionDock
  protected detail                                       [B,24,4,H]

ObjectPolicyPlanDeltaBank
  protected base + precision/effect/temporal/state-change [B,24,4,H]

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
| S | T5, state/executed history, one public scene value and K/type ObjectFactSet innovations | frame progress, phase label, noisy action, Teacher, K copies of public content |
| W | one public ObjectFactSet value plus K innovations/transport, S-owned typed relevance, one typed clean coarse action intent | raw semantic/appearance/geometry reread, target/noisy action, proposal, Teacher, free W residual |
| P1 | completed progressive chart, S, clean action bases | global-K value, W, proposal, Teacher, second visual read |
| P2 | completed P1 fact, supervised per-type W fields, typed S keys, noisy-action query | RGB/DINO reopen, camera-expanded W, free W hidden |
| P3 | cached P1 precision innovation, consequence, S temporal innovation, noisy-action query | Teacher, RGB/DINO, proposal, free W carrier, duplicated factual lane |
| transition source | exact completed G3 rollout | W target, proposal, noisy action, Teacher |
| transition dynamic | source, shared V120 seed, plan | target action, Teacher, future proposal |
| bottom | consequence, four active P3 lanes, transition, seed, layer contracts | RGB/DINO, Teacher, duplicate W/P base |

## Loss and optimizer ownership

- Physical V120 action flow matching remains dominant.
- The global grounder owns exactly one observed-current dense reconstruction
  MSE; the existing intent-structure ledger applies its fixed 0.25 internal
  coefficient. Separate typed compatibility changes the physical assignment,
  not the number of G objectives. No diversity, entropy, prototype or
  typed-consistency loss is introduced.
- Teacher successor is the uniform interval mean of
  `matched + null_probability * current_reference`; semantic delta is exactly
  successor minus current reference. Transport/covariance are uniform means of
  raw posterior moments formed from same-camera
  `future_coordinate-current_camera_coordinate` displacements.
  Reliability/entropy do not shrink targets or masks.
- `current_loss_support [B,K,C,1]` is the detached physical
  `ObjectFactSet.camera_validity`; it owns future and typed-S supervision after
  camera reduction. Assignment `camera_evidence_mass` remains camera
  reduction/audit evidence and cannot mask training loss. `object_existence`
  is detached and belongs only
  to optional online candidacy. `future_selector_validity [B,4,K,1]` belongs
  only to online P2 routing.
- Action, future, flow geometry, intent scaffold, history proposal and
  execution-value external weights are unchanged from the recovery reference.
- S has no training-only recognizer hidden. Its observable condition
  innovation owns the public future-state head; typed semantic/status/
  transport supervision decodes the exact pre-W field through the same
  projections and heads used by W. Their losses reuse the existing intent
  scaffold; no external weight is added. The exact online CoarseAction tensor
  consumed by W alone owns future-action regression and is not recomputed for
  loss attachment.
- Every trainable parameter has exactly one optimizer owner. Ordinary bias,
  LayerNorm, top/controller/decoder parameters use AdamW decay 0.01. Only
  explicitly named scale-invariant contraction basis/depth coordinates are
  no-decay.
- Gradient lifecycle is:

  ```text
  finite check -> gradient_raw
  -> non-controller bottom.decoder local clip 1.0 -> gradient_postlocal
  -> non-controller global clip 1.0
  -> execution controller independent clip 1.0 -> gradient_postglobal
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
- Schema29 parameter counts are measured after implementation and must be
  reported by the launcher. The verified default graph has `172,151,237`
  parameters, of which `155,756,916` are trainable. Its direct-child inventory
  is observation `13,543,661 / 6,895,950`, top
  `81,119,243 / 81,020,939`, history proposal
  `10,014,727 / 10,010,631`, factual P1 `3,612,429 / 2,823,949`, transition
  `8,027,785 / 7,895,049`, and bottom `55,833,392 / 47,110,398`
  (total/trainable). Schema28's top-only delta came from typed clean-action
  reads, typed W block traversal and three independent P2 selectors; removal
  of private G decoder heads, the factual pseudo-lane and several duplicated
  P3 projections offsets part of that increase. Bottom and the exact P1 reader
  remain unchanged. Schema29 removes the `1,536`-parameter outer type selector
  and adds a `131,584`-parameter rank-64 bias-free contrast residual plus
  LayerScale, for a net top increase of `130,048` parameters.
- Active manifest identity:

  ```text
  schema:       29
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_bank
  top:          conditional_k_public_private_grounding_zero_centered_intent_single_typed_ingress_w_per_type_p2_anchored_contrast_fusion_consequence_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_four_active_plan_lanes_exact_g3_anchor_transition_evidence_mmdit_dense512_execution
  training:     v120_mirrored_physical_flow_observed_current_grounding_physical_camera_loss_support_shared_w_field_targets_exact_teacher_event_boost_v120_decay_three_owner_clip
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

The local mainline/audit suite covers full forward/backward,
G1/G2/G3 ordering and N=49
rematerialization, forbidden G conditions, exact P1 axes/microgrid,
chunked/unchunked P1 output and gradients, Teacher isolation, object/camera
permutations, per-type S perturbation locality, exact-null goal/typed values,
typed-owner relabeling equivariance, public-target gradient isolation,
typed CoarseAction ownership, W1/W2 typed-state traversal, public-W inability
to synthesize a zero typed field, per-type P2 locality, anchored type-contrast
fusion/all-null zero, physical camera loss support, four active P3 sources,
same-camera Teacher geometry, object geometry, neutral effect, endpoint lifecycle,
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
OUT_DIR=runs/schema29_complementary_p2_smoke \
nohup bash scripts/smoke_mainline.sh > schema29_complementary_p2_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema29_complementary_p2_b8 \
nohup bash scripts/train_mainline.sh > schema29_complementary_p2_b8.log 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema29_complementary_p2_b8 \
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
