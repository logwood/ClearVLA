# Current ClearVLA Architecture Contract

Updated: 2026-08-28

This is the compact source of truth for the active independent mainline.
Experiment labels never select model semantics. Historical evidence lives in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`; only still-open work belongs in
`CURRENT_MAINLINE_ISSUES.md`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        25
behavior reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
source reference:       .audit/v120_exact_source_0b92d359/
release status:         Schema25-R2 WG01/P202/GRIP02 locally closed; 170 retained tests plus two R2 cadence/reachability guards, a fresh production-dimension one-batch CPU BF16 smoke and five-step deployment guard pass; no R2 GPU smoke or formal behavior run yet
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0, two adjacent learned flows
formal language:        precomputed 4096-wide T5 .pt required
bottom:                 V120 seed/transition/CVAE/workspace/Evidence MMDiT/execution
long launcher:          scripts/train_mainline.sh (batch 8, workers 4)
smoke launcher:         scripts/smoke_mainline.sh (batch 1, workers 0)
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only; no optimizer/schedule/RNG load or checkpoint write)
resolved config:        configs/mainline/object_intent_dynamics_323.json
```

> **The executable source is the exact Schema25 replay base plus completed
> R1a/G-01, R1b/G-02, R1c/S-01,S-02, R1d/W-01,W-02, R1e/P1-01,
> R1f/P2-01, R1g/P3-01,B-01, LC-01, R1h/N-01,D-01 and the three
> R2-WG01/P202/GRIP02 structural units; R2 is not yet
> behavior-released.** The untouched R0 fingerprint,
> selected cross-version units and implementation gates live in
> `docs/research/auxiliary/SCHEMA25_R0_BASELINE_FINGERPRINT.md`,
> `ARCHITECTURE_REPLAY_SOURCE_UNITS.md` and
> `SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md`. R1a repairs the independently
> confirmed G3-to-transition handoff. R1b repairs conditional-K and
> reconstruction ownership without changing the Schema25 binder inputs or
> parameter inventory. R1c removes the duplicate typed CoarseAction-to-W path
> and expresses the unchanged Schema25 relevance value as exact interval-common
> plus residual coordinates. R1d gives those coordinates one causal W owner,
> retains physical camera identity through future geometry, and removes online
> predicted-status authority. R1e separates cached factual detail from the
> noisy-action/time-dependent P1 policy residual and restores that raw residual
> only at its named dynamic consumers. R1f performs semantic K and geometry
> K*C selection independently inside every interval, then removes I through
> one no-null physical terminal per type before their raw complementary sum.
> R1g removes the three optional aliases of already-protected values and gives
> the remaining temporal and state-change innovations separate shared-parameter
> Q+null decisions at bottom.
> LC-01 then deletes two audited exact-zero layer-contract trajectory aliases
> and their frozen readouts while preserving every live contract/decoder
> tensor, optimizer owner and the fresh-run RNG stream.
> R1h keeps the same measures and owners while making the four active address
> variance-to-standard-deviation paths zero preserving, retaining producer-owned
> FP32 probability/log views through G-to-P2, and observing only live source
> tensors and finite raw-gradient spikes.
> R1 has since completed its first formal run. R2-D01 adds observation and
> R2-A01 adds evaluation-only matched P2 value counterfactuals plus gripper
> attribution. Neither changes training, checkpoint state, primary deployment
> behavior or losses. The completed A01 replay shows that semantic intervals
> `2/3` are strongly useful to far action, geometry values have near-zero
> action effect at their learned R1 scale, and both deployed gripper branches
> fail before their fixed blend. R2 now makes the three source-bounded repairs:
> target-scale-covariant camera-transport supervision, independent exact-copy
> spatial/terminal P2 query owners, and one exact-zero continuous
> gripper-private state for the deployed value/delta heads, followed by a
> codec-decoded absolute/delta event boundary. Event supervision reaches the
> private state through the physically consumed value/delta branches; there is
> no hidden-state event bypass. It adds no P3, sampler, checkpoint-selector,
> time schedule,
> hard event gate or loss-weight bundle. Local implementation guards pass; no
> R2 behavior result is implied before the GPU smoke and formal run.

> **Replay scope lock:** R1 is assembled as reversible semantic units in the
> adopted order. A later unit cannot be implemented until its complete
> producer/consumer/loss/runtime/checkpoint worksheet closes. Historical
> schemas are donor coordinates, not phases or whole commits to replay.

The active graph lives in `clearvla/mainline/`. It does not dispatch through
the V39 trainer/runtime/trunk or a V-numbered capability branch. The manifest
owns serialized graph identity; typed interfaces and executable checks own
shape, dtype, provenance and zero semantics. Do not add a version-wide
`_validate_vXXX_*` contract.

The active replay retains the controlled Schema24 V120 fidelity recovery,
including its flow-time, endpoint-head, Teacher association algebra,
support/selector split, non-finite sentinel and the following four
source-audited boundaries:

- literal G block -> progressive updater alternation at stages 1, 2 and 3;
- V120 P1 factorization with 24 factual queries, N=49 and a 3x3 microgrid;
- one dense global-object reconstruction objective;
- camera-specific online future geometry, retaining physical camera identity
  through W, Teacher targets, future losses and the P2 geometry consumer.

It also retains V120 AdamW decay and decoder-local then global clipping.
Schema25 makes one bounded S ownership change:

- S separates its supervised public interval carrier from optional typed
  relevance and preserves `[interval,K,type]` until named consumers;
- semantic, appearance and geometry each compare only with a fixed zero null;
- CoarseAction and W consume S-owned docks and no longer reread/reselect raw
  typed facts through their own learned-null routers;
- the Schema25 base initially leaves P1, P3, transition and bottom internals
  unchanged; R1d makes the minimum camera-aware P2 adaptation, R1e repairs P1
  carrier ownership, R1f closes the P2 spatial/physical terminal, and R1g
  closes P3/bottom alias competition below.

R1a/G-01 makes one additional source-local repair on that base:

- P1 and the controlled transition now consume the same exact completed G3
  rollout view `[B,4*C*8*8,H]`;
- transition no longer reconstructs four anchors from an anchor-averaged
  public chart and no longer owns `interval_identity`;
- the handoff adds no cast, clone, detach, projection, normalization, gain,
  floor or amplitude budget.

R1b/G-02 makes one additional grounder-local ownership repair:

- the detached current DINO chart is the reconstruction target only; the
  online candidate chart remains the completed G3 local content;
- the physical Schema25 K+null binder still decides real-versus-null mass,
  while G3 changes only `P(K | real)`;
- reconstruction assignment is conditional K times the retained local prior
  and observable validity, so learned null cannot switch off this objective;
- the existing zero-initialized slot content residual is folded into the one
  exported `ObjectFactSet.content` consumed by S, W and detached Teacher;
- the existing coordinate decoder remains a shared K-independent spatial term.
  No decoder, gain, floor, content field or loss was added or removed.

R1c/S-01,S-02 makes one S/W-boundary repair without changing the selector:

- the Schema25 bounded-cosine relevance score, temperature, physical K/type
  axes, zero semantics and typed policy component are numerically unchanged;
- relevance mass and value are stored as their four-interval mean plus signed
  zero-sum interval residual, so `common + residual` reconstructs the former
  tensor without a gain, floor, clamp, detach or new parameter;
- `ActionIntentDock` and CoarseAction contain only public interval,
  observable-history and public-object context. Typed evidence therefore
  reaches W through `WorldIntentDock` once rather than again through action
  tokens;
- factual P1 and P3 retain their reduced policy context. P2 additionally reads
  the existing typed common/residual S metadata through `PolicyIntentDock`,
  but only after a W-owned spatial posterior has selected it. At the R1c
  boundary, pre-R1d W
  reconstructed its former full typed source once at `_base`; R1d below keeps
  the common/innovation coordinates distinct without adding a second path.

R1d/W-01,W-02 replaces the pre-R1d W mechanics and future ABI:

- W1 processes the protected `[B,K,3,H]` typed common exactly once, then owns
  only the two near innovations. W2 reads completed common/near state and owns
  only the two far innovations; it cannot rewrite common or near;
- common-to-innovation, generic and appearance conditioning use the
  parameter-free zero-preserving relation `x + x * tanh(c)`. A zero typed
  owner therefore remains exact zero and no conditioner becomes a free W
  value;
- appearance conditions semantic state. Geometry is conditioned separately by
  each existing current-camera transport prior and remains
  `[B,4,K,C,2|3]`; covariance is FP32 PSD and may approach exact zero;
- `FutureObjectDynamics` contains semantic delta/successor, camera transport
  and covariance, and observable current chart/camera support only. Predicted
  visibility, persistence, uncertainty, reliability, validity, future address
  and reduced object coordinates are absent;
- Teacher retains the Schema25 candidate-plus-null row softmax, but its
  post-association displacement moments and null-identity allocation remain
  per camera. P2 consumes camera coordinate, covariance and transport before C
  is reduced and uses only current observable support;
- R1d intentionally leaves the inherited flattened `[interval,K]+null`
  terminal and semantic-versus-geometry type softmax as P2-01 debt; R1f below
  closes that debt without changing the W field.

R1e/P1-01 separates the static and dynamic P1 owners without changing their
producers:

- the exact cached `FactualPrecisionDock.protected_detail` remains
  `CompletedP1PolicyState.factual_base`; the per-ODE
  `updated_canvas-canvas` write remains
  `policy_query_residual`, with no eager combined factual alias;
- P2 alone forms `action_query + factual_base + policy_query_residual` through
  `P2QueryDock`. Protected consequence starts from `factual_base`, so the
  dynamic write cannot re-enter the graph under a factual identity;
- `ObjectPolicyPlanDeltaBank.protected_policy_precision` carries the exact raw
  dynamic residual. At the R1e boundary optional P3 projections still received
  static factual and consequence aliases; R1g below removes that P3-01 debt;
- the controlled transition adds the raw residual once in its existing
  terminal-normalized action operand. Bottom reads the same carrier once with
  its existing no-null basis reader and existing optional-ingress scale;
- no reader, parameter, buffer, persistent key, RMS contract, LayerScale,
  learned gain, learned null or second bottom scale was added.

R1f/P2-01 separates spatial selection from the physical interval terminal:

- semantic selection normalizes K independently for every `[B,T,Q,I]` row;
  geometry consumes transport, FP32 covariance and current camera coordinate
  while normalizing K*C independently for every interval;
- `SelectedIntervalEvidence [B,T,Q,I,2,H]` retains interval and type until the
  terminal and preserves an exact selected common-plus-innovation identity;
- the same W-owned spatial posterior selects existing typed S metadata. S can
  condition the selected W key only through
  `key + key * tanh(bounded(S))`; it cannot create support, value, spatial
  selection or an independent interval vote;
- semantic and geometry each own a no-null four-physical-interval posterior.
  All-invalid observable support is finite exact zero;
- their raw latent contributions are added without a type softmax, divisor or
  per-type gain. This `+` is the adopted complementary latent fusion operator,
  not a physical-units claim. The existing caller-owned P2 RMS contract is
  still the only outer amplitude boundary.
- R2-P202 retains that exact axis/support/value algebra but assigns spatial
  K/K*C selection and physical-I termination separate bias-free query
  projections. `terminal_query` is an exact construction-time copy of
  `source_query`, consumes no inherited initialization RNG and is used only by
  the four-interval terminal. No time prior, interval target, null or new gain
  is introduced.

R1g/P3-01,B-01 removes duplicate P3 value owners and cross-lane competition:

- `ObjectPolicyPlanDeltaBank` contains the protected consequence, the exact raw
  dynamic P1 residual, and only two optional innovations: temporal and
  observable state-change;
- temporal reads S temporal context plus the P2 effect/interaction innovation,
  conditioned by action. State-change multiplies its S-private evidence by an
  action/temporal condition. Both paths are bias-free and preserve exact zero;
- optional factual, static-precision and effect projections are absent. The
  inherited `0.05` state-change multiplier and `sqrt(2)` divisors are removed
  without replacement gains, floors or quotas;
- bottom invokes one shared Q+zero-null reader separately for temporal and
  state-change, so one lane cannot renormalize the other. Their raw reads add,
  then the no-null dynamic-precision read joins before the one inherited fixed
  optional scale;
- protected consequence remains a separate no-null read outside that scale.
  No aggregate lane-sum contract, lane-specific capacity or second bottom
  scale is introduced.

LC-01 removes the two V120 layer-contract trajectory formulas after proving
their exact-zero VJP and exact intervention invariance. The two independent
terminal depth adapters remain live on rollout/state rows and still supply the
event input from rollout delta. Their frozen trajectory action/motion readouts
and trajectory-shaped compatibility outputs are absent. P2, controlled
transition and bottom remain the actual dynamic-P1 consumers.

R1h/N-01,D-01 changes numerical representation and observation only:

- the four live address variance consumers use
  `v / (sqrt(v + epsilon^2) + epsilon)`, with the existing grid/radius scale;
  exact-zero variance stays exact zero and its reverse gain is finite;
- G2 produces FP32 typed slot log probabilities, G3 applies its bounded
  residual in log space, and the same probability/log measure passes through
  Local/Dense/Object/Future boundaries to P2 without a probability floor;
- unsupported and all-invalid rows remain boolean-authorized exact zero; a
  legacy fixture without producer logs retains an explicit FP32 fallback and
  cannot resurrect a zero-prior candidate;
- backward hooks copy detached source-tensor RMS values and return the incoming
  gradient unchanged. Finite spike attribution runs only above its observation
  threshold and before the unchanged local/global clipping lifecycle.

No reliability/status/content carrier, learned scale, route, loss, support
floor or parameter was imported from Schema38/39 with these support mechanics.

No block, external loss weight, gain, quota, hard gate, entropy target or
capacity was added. R1d removes three W status heads and one P2 status
projection; R1e has zero parameter/state delta. R1f removes the P2 type query
and adds only the missing geometry key plus two type-specific projections of
existing S route metadata. R1g removes six duplicate P3 projections and sixteen
inactive optional source-key rows. Exact resume across the R1e/R1f or R1f/R1g
state ABI is intentionally incompatible and no migration shim exists. R1h has
zero parameter/state/RNG delta, but its probability/log and runtime component
ABI changes the manifest, so an older exact checkpoint is also rejected.

The post-epoch-1 source/log closure audit then corrected four remaining
fidelity defects without changing the G/S/W/P or bottom topology:

- active V120 pre-G/address/future-query parameters are trainable again; only
  the unconsumed object-intent G3 generic route query remains frozen;
- Teacher transport/covariance form displacement moments inside each camera,
  so a camera-mass change cannot invent motion for a static object. R1d keeps
  those moments camera-resolved rather than immediately reducing them;
- the global-K binder no longer adds the public chart equally to every private
  candidate key;
- validation diagnostics are spread across the full loader and proposal,
  sampling and exact V120 execution ablations have separate coverage.
  Primary deployment noise is restored to V120's deterministic per-batch
  stream (`37237 + one-based batch index`), and every ablation reuses that
  exact physical noise.

These corrections and the R1 ownership/ABI changes alter the source and
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
       zero-preserving finite-slope variance-to-standard-deviation conversion
    -> G2 DiT -> progressive update stage 2: rematerialize N=49 fine candidates
       FP32 typed local-slot probability + producer log probability
    -> G3 DiT -> progressive update stage 3: bounded log-space owner correction
    -> completed camera x 8x8 x local-M GroundedFactSet
    -> competitive global K+null DenseObjectGrounder
       physical real/null mass + conditional-K-only G3 correction
       FP32 observable object/camera probability + producer log probability
       detached observed-current-DINO reconstruction target
    -> ObjectFactSet with reversible K <-> chart correspondence
       one exported K content value shared by reconstruction/S/W/Teacher

T5 + observable state/executed-action history + ObjectFactSet
    -> StatelessObjectIntentOrganizer S
       protected goal/history/public-object values
       supervised public interval carrier [B,4,H]
       per-type fixed-zero relevance [B,4,K,3,*]
       consumer-specific policy interval context
       24 temporal queries and state-change evidence

ObjectFactSet public content/transport + S-owned WorldIntentDock
    + causal clean CoarseActionIntent from ActionIntentDock
    -> W1: typed common once, then 4-8 and 8-16 near innovations
    -> W2: 16-32 and 32-48 far innovations, read-only over completed W1
    -> semantic FutureObjectDynamics [B,4,K,D]
       plus camera transport/covariance [B,4,K,C,2|3], no predicted status
       copied current observable object/camera probability + log probability

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
    -> CompletedP1PolicyState
       factual_base = cached protected detail
       policy_query_residual = updated canvas - input canvas

action query + factual_base + policy_query_residual
    -> exact P2QueryDock combined query
    + FutureObjectDynamics + S
    -> semantic K and geometry K*C selection independently within each I
       using current chart/camera availability as the only physical support
    -> SelectedIntervalEvidence [B,T,Q,I,semantic|geometry,H]
    -> S-conditioned W key, one no-null physical-I terminal per type
    -> raw semantic + geometry latent sum
    -> one inherited caller-owned P2 RMS contract
factual_base + P2 effect
    -> zero-preserving protected consequence
P2 effect + factual/effect interaction + S temporal context + action condition
    -> optional P3 temporal innovation
S state-change evidence * action/temporal condition
    -> optional P3 state-change innovation
raw protected_policy_precision = policy_query_residual

one shared V120 action/context seed
    -> noisy-action query shared by P2/P3/transition/bottom
    -> current state, causal state history, compressed executed history

exact completed G3 rollout shared with static P1
    -> static 512-row ControlledTransitionSource, once per observation
noisy action + protected consequence + raw policy residual
    + V120 learned neutral + plan/history
    -> dynamic 512-row ControlledTransitionState, every dynamic forward

transition + shared seed
    -> two independent terminal policy-depth layer contracts
    -> CVAE/workspace/EvidenceViewAdapter
protected consequence
    -> separate no-null protected-detail basis read
temporal and state-change P3 lanes
    -> separate shared-parameter Q+zero-null reads
    -> raw optional-lane sum
raw policy residual
    -> separate no-null basis read
    -> join optional source read before the one inherited fixed scale
all bottom carriers
    -> three Evidence MMDiT blocks
    -> ordered low-rank contraction and execution-value controller
    -> 18-D physical velocity plus codec-decoded event and motion heads
```

The history-action proposal remains a supervised auxiliary prediction. Its
future proposal tokens do not enter G/S/W/P, transition or bottom. The
separately compressed executed-action history remains an observable condition
in the shared seed. Generic trajectory/workspace ingress is algebraic neutral;
protected consequence is written once through its no-null ingress. The raw
dynamic P1 residual is read separately without a null and joins the two
independently selected P3 innovations only at the retained optional-ingress
scale.

Training-only graph:

```text
future DINO supports
    -> FP32/no-grad Teacher association, once per training batch
    -> four semantic and camera-resolved FutureObjectDynamics targets
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
   physical K=4 plus null competition. G3 preserves the parent's real/null
   mass and refines only conditional K. Reconstruction assignment is
   `P(K|real) * local_prior * observable_validity`; its target is detached
   current DINO over observed cells, and its sole K-specific value is exported
   `ObjectFactSet.content`. The public chart remains outside private candidate
   competition and is never a second object value. Typed reads can reweight
   only inside physical K support.
6. Learned flow is a continuous source-relative prior, never a nonzero quota.
   Flow warp/cycle/smoothness/uncertainty/refinement keep explicit units.
7. S reads full T5, observable state/action history and typed G facts. It does
   not read frame progress, phase labels, noisy action or future Teacher. Its
   public carrier is supervised separately from optional typed relevance.
   Semantic, appearance and geometry retain real K/type axes, each owns a
   fixed-zero null comparison, and only `WorldIntentDock` may deliver their
   common/residual values to W. CoarseAction has no typed field. Factual P1
   retains only reduced S context; `PolicyIntentDock` exposes the existing
   typed common/residual metadata solely so P2 can select it with W's spatial
   posterior after K/C selection authority has already been fixed.
8. W1 owns typed common exactly once and the two near innovations. W2 may read
   completed common/near state but writes only the two far innovations. Every
   public interval is one processed common plus its matching processed
   innovation. All generic/action/goal/appearance/camera conditions are
   zero-preserving and cannot synthesize a W value. The only W value below W
   is directly supervised `FutureObjectDynamics`; no public or private free W
   carrier crosses into P.
9. P1 owns 24 factual queries before action-basis organization, four factual
   glimpse types, the complete N=49 posterior and a real 3x3 microgrid.
   Global-K is not a P1 axis. `FactualPrecisionDock` is a parameter-free
   boundary containing only the already-computed protected detail; it is not a
   replacement reader or an extra bottleneck. `CompletedP1PolicyState` keeps
   that static fact separate from the noisy-action/time-dependent
   `policy_query_residual`. Only P2 may materialize their three-term sum with
   the action query; no eager combined factual alias is stored.
10. P2 geometry remains camera-specific until its named consumer:

    ```text
    transport_mean/covariance       [B,4,K,C,2|3]
    camera_coordinates              [B,K,C,2]
    chart_availability              [B,K,1]
    camera_chart_availability       [B,K,C,1]
    ```

    W obtains C only by conditioning typed geometry on the existing physical
    camera transport prior; it cannot predict once and duplicate a fake camera
    axis. Teacher targets, future losses and P2 preserve C. P2 consumes the
    covariance metric and transport in its geometry K*C posterior before
    reducing C. Semantic K and geometry K*C selection retain I; the named
    terminal then normalizes only the four physical intervals independently
    per type, with no null. Spatial and terminal queries have independent
    trainable owners but start as an exact functional identity. The same
    spatial posterior selects S metadata, which can condition only a nonzero W
    key. Semantic and geometry add before one outer contract and never compete.
    No predicted status or predicted validity controls support.
11. Neutral effect is algebraically neutral:

    ```text
    effect = 0
    interaction = 0
    protected_consequence = factual_base
    ```

12. P3 owns exactly two optional lanes: temporal and state-change. Protected
    consequence and the exact raw dynamic P1 residual remain separate carriers;
    no optional factual, static-precision or effect alias exists. Each optional
    lane has one private zero-preserving operand and cannot reopen vision,
    consume a free W carrier or reproject the complete factual consequence.
13. Transition static/dynamic frequency is model semantics: its exact final
    G3 rollout source builds once; real-versus-neutral coefficients read
    current noisy action, protected consequence and raw P1 policy residual at
    every dynamic forward. The source retains the real anchor/camera/xy rows
    and may not recreate them from a reduced chart.
14. Bottom source count/order/value semantics follow V120. Do not remove CVAE,
    workspace, P1/P2 contracts, Evidence MMDiT, capacity or execution to reduce
    memory or simplify the mainline. Protected consequence and dynamic P1
    precision use separate no-null basis calls. Temporal and state-change use
    separate invocations of one shared Q+zero-null reader; they never share a
    probability simplex. Their raw reads add before dynamic precision joins the
    one inherited fixed optional scale. Consequence stays outside that scale;
    no second scale or aggregate magnitude contract exists. At the final output
    boundary, a bias-free zero-initialized bounded multiplicative owner forms
    one continuous gripper-private state. Deployed gripper value/delta read
    that state; the policy then decodes their physical absolute and
    adjacent-delta coordinates into the supervised event boundary. Arm, four
    auxiliary gripper coordinates and motion retain the base action read.
    Event logits never gate or otherwise enter the physical field.
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
  typed common mass / value                              [B,K,3,1|R]
  typed interval-residual mass / value                   [B,4,K,3,1|R]
  typed policy components                                [B,4,3,H]
  temporal queries / state-change evidence               [B,24,H] / [B,H]

Consumer views
  ActionIntentDock (typed-free public action context)
  WorldIntentDock (typed common/residual W ingress)
  FactualIntentDock (named reduced factual S context)
  PolicyIntentDock (reduced P3 context plus existing typed P2 metadata)

FutureObjectDynamics
  current reference                                      [B,K,D]
  successor / semantic delta                             [B,4,K,D]
  transport / FP32 PSD covariance                        [B,4,K,C,2|3]
  current chart availability                             [B,K,1]
  current camera coordinates / availability              [B,K,C,2|1]

ObjectTopTrainingTargets
  current loss support                                   [B,K,C,1]

FactualPrecisionDock
  protected detail                                       [B,24,4,H]

CompletedP1PolicyState
  factual base / live policy-query residual              [B,24,4,H]

P2QueryDock
  action query / factual base / policy-query residual    [B,24,4,H]

SelectedIntervalEvidence
  key/value/common/innovation/S metadata                 [B,24,4,4,2,H]
  observable semantic/geometry support                   [B,4,2]

ObjectPolicyPlanDeltaBank
  protected base / protected policy precision            [B,24,4,H]
  temporal / state-change optional innovations            [B,24,4,H]

V120SeedContext
  state / state history / compressed executed history    [B,1|3|7,H]

ControlledTransitionSource / State
  static selector / dynamic selector and value            [B,512,H]

GripperPrivateState
  action + action * tanh(bias_free_gate(norm(action)))     [B,24,H]
  consumers: deployed gripper value/delta heads

DecodedGripperEventBoundary
  codec-decoded gripper absolute + adjacent delta              [B,24,2]
  consumer: supervised final event head (reverse path via value/delta)
```

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| G | current DINO/raw history, coordinates, learned flow, current state | T5, action history, proposal, noisy action, Teacher |
| global grounder | completed G3 chart/typed local candidates; detached current DINO and observed mask for its sole reconstruction loss | S, W, noisy action, future Teacher data |
| S | T5, state/executed history, typed ObjectFactSet | frame progress, phase label, noisy action, Teacher |
| W | public ObjectFactSet conditions, existing camera transport prior, S-owned typed common/residual through WorldIntentDock, one typed-free clean coarse action intent | raw typed-fact reread, second typed action path, target/noisy action, proposal, Teacher, predicted status/support, free W value |
| static P1 | completed progressive chart, S, clean action bases | global-K value, W, proposal, Teacher, noisy action/time, second visual read |
| dynamic P1 | action query, Euler time, cached factual base | vision reopen, W, proposal, Teacher, factual relabeling of its live residual |
| P2 | exact three-term query, supervised semantic/camera W field, current chart/camera support; S metadata only after W spatial selection | RGB/DINO reopen, predicted status/support, S-owned K/C/time vote, type competition, free W hidden |
| P3 | protected consequence carrier, P2 effect/interaction innovation, S temporal/state-change context, noisy-action query; raw dynamic residual only as protected policy precision | complete factual-consequence reprojection, optional factual/static-precision/effect aliases, Teacher, RGB/DINO, proposal, free W carrier, dynamic residual projection into optional lanes |
| transition source | exact completed G3 rollout view shared with P1 | W target, proposal, noisy action, Teacher |
| transition dynamic | source, shared V120 seed, protected consequence, raw P1 policy residual, plan | target action, Teacher, future proposal |
| bottom | consequence, raw P1 policy residual, two independently selected optional P3 lanes, transition, seed, terminal rollout/state/event layer contracts | joint cross-lane simplex, RGB/DINO, Teacher, duplicate W/P base, layer-contract P1/P2 trajectory aliases, new dynamic gain/null |

## Loss and optimizer ownership

- Physical V120 action flow matching remains dominant.
- The global grounder owns exactly one observed-current-DINO reconstruction
  MSE. Its assignment is conditional-K, local-prior and observable-validity
  mass; its only K-specific value is exported object content. The existing
  intent-structure ledger applies its fixed 0.25 internal coefficient. No
  prototype, masked-completion or typed-consistency head remains.
- Teacher successor is the uniform interval mean of
  `matched + null_probability * current_reference`; semantic delta is exactly
  successor minus current reference. Per-camera transport/covariance are
  uniform interval means of raw posterior moments formed from same-camera
  `future_coordinate-current_camera_coordinate` displacements, with null mass
  assigned the identity displacement independently per available camera.
  Reliability/entropy do not shrink targets or masks.
- `current_loss_support [B,K,C,1]` owns semantic, camera-transport and
  camera-covariance losses; only the recognizer receives its detached camera
  reduction. Online P2 support is the current chart/camera availability carried
  by `FutureObjectDynamics`; there is no predicted-selector validity.
- Future semantic objectives retain their exact-zero raw/normalized/direction
  row loss. Camera transport common/innovation use raw-coordinate SmoothL1 as
  the sole active measure, redistributed by detached inverse-square target
  scale weights whose supported mean is one; covariance remains raw-coordinate
  and per camera. Detached unweighted-raw, normalized and direction transport
  audits never enter backward. The internal
  `0.55/0.15/0.05` semantic/transport/covariance coefficients and outer future
  weight are unchanged. No successor duplicate or status objective remains.
- Action, future, flow geometry, intent scaffold, history proposal and
  execution-value external weights are unchanged from the recovery reference.
- The whole-segment recognizer supervises only S's public interval carrier.
  Typed relevance is trained through future W and the factual/P2/P3/final
  action paths. The typed-free coarse-action loss does not reach its selector;
  no public future target, entropy or usage loss directly trains it.
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
  times `[0,.2,.4,.6,.8]`, then once at `1.0` for the codec-decoded event and
  motion heads only. The endpoint call cannot change the integrated action.
- Teacher builds once per training batch and zero times in deployment.
- P1 N=49 queries use the V120 query budget/checkpoint configuration.
  Chunked and unchunked outputs and parameter gradients must be equivalent.
- Startup writes a per-module parameter inventory. Counts are measured, never
  hard-coded into the contract; any difference from V120 must name the removed
  and restored owners.
- The untouched Schema25 R0 configuration measures `169,981,895` total and
  `153,587,574` trainable parameters. R1a, R1b and R1c measure `169,979,847`
  total and `153,585,526` trainable parameters; the exact `-2,048` delta is the
  R1a removal of trainable `transition.interval_identity`. R1b retains the
  grounder's 4,007,936 parameters and 17 optimizer tensors exactly, and R1c
  adds no module or state key. R1d measures `169,976,772` total and
  `153,582,451` trainable parameters. Its exact `-3,075` delta is three W
  status heads (`-1,539`) plus the P2 status projection and third type-query
  row (`-1,536`). R1e is parameter-free and retains the exact R1d inventory
  and ordered state-key sequence. R1f adds a net 32,768 trainable parameters:
  a geometry source-key projection and two type-specific S route projections
  replace the removed type-query projection. The R1f model has 170,009,540
  total / 153,615,219 trainable parameters, 1,408 parameter tensors, 1,070
  trainable/optimizer tensors, 23 optimizer groups and 1,414 state-key names.
  W measures 9,229,827 parameters / 26 tensors; P2 measures 1,608,707 / 11.
  R1g removes six H-to-H P3 alias projections (`-1,572,864`) and shrinks the
  bottom optional reader's serialized source-key rows from `5*Q` to `Q`
  (`-512`). The active R1g model has `168,436,164` total / `152,041,843`
  trainable parameters, 1,402 parameter tensors, 1,064 trainable/optimizer
  tensors, 23 optimizer groups and 1,408 state-key names. P3 measures
  1,572,864 parameters / 6 tensors. LC-01 then removes 23,590 frozen
  trajectory-only readout parameters and 16 parameter/state tensors. The
  active model has `168,412,574` total / `152,041,843` trainable parameters,
  1,386 parameter tensors, 1,064 trainable/optimizer tensors, 23 optimizer
  groups and 1,392 state-key names. The ordered state-key-name SHA-256 is
  `be7b4b58a8e2ec25c1e3b5c455f303a0954d20a984201173b5de12d2b1f14a20`.
  An independent seed-0 R1d/R1e construction comparison produced identical
  canonical full-state tensor digest
  `9793ea81a3b1173c7569300bc74a31f462c2e792744d0f2299d5ccdfd3ec5ba7`
  and identical post-construction CPU RNG digest
  `8670db504a2bb9d1e15f1d87977890e5006f320ab4657a52e9963ea674c67250`.
  Relative to the completed Schema24 graph,
  the exact `-12,734,208` trainable delta is fully accounted for: S removes
  three duplicate `_CrossRead`s plus one shared learned-null router and adds
  three route-width relevance projections plus three temperatures
  (`-6,308,093`); CoarseAction removes three duplicate `_CrossRead`s and one
  learned-null router (`-6,357,248`); W removes one learned-null router
  (`-65,792`) and three status heads (`-1,539`); P2 removes its status
  projection plus one type-query row (`-1,536`). Through R1f, bottom and the
  exact P1 reader parameter inventories are unchanged; R1e changes only
  parameter-free runtime carriers and existing-consumer wiring. R1f
  intentionally changes P2 state names and fresh-run RNG. R1g consumes the
  removed historical initialization draws, so every retained and downstream
  fresh-run tensor keeps the R1f stream. Its seed-0 post-construction CPU RNG
  SHA-256 remains
  `d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
  LC-01 uses initialization-only temporary historical readouts, so its 46
  retained layer-contract state tensors, all 268 decoder state tensors and
  this RNG digest remain byte-identical while the discarded readouts own no
  runtime or checkpoint state. R1h registers no parameter, buffer or random
  draw and retains these counts, the ordered key digest, both retained tensor
  digests and the construction RNG exactly. The first R2 structural inventory
  added exactly two `top.effect_reader.terminal_query.*.weight` tensors and
  one `bottom.decoder.velocity_head.gripper_gate.weight`, all bias-free H-to-H
  matrices (`169,199,006` total / `152,828,275` trainable parameters, 1,389
  parameter tensors, 1,067 trainable/optimizer tensors, 23 optimizer groups
  and 1,395 state keys). The completed codec-closed/event and
  typed-consequence boundary then adds one exact-copy
  `top.consequence.geometry_interaction` matrix, removes the hidden bottom
  event head and adds a zero-initialized `decoded_gripper_event_head` (`2 -> 3`).
  Its net delta from that first R2 inventory is `+259,590` parameters, `-1`
  parameter tensor and `-1` state key, with no optimizer-group change. The
  current R2 model has `169,458,596` total / `153,087,865` trainable
  parameters, 1,388 parameter tensors, 1,066 trainable/optimizer tensors, 23
  optimizer groups and 1,394 state-key names. Its ordered state-key-name
  SHA-256 is
  `384bf6aa4f765382f3d7b4251f0b70f53fe233d3a86090f8ea2bdad6d886d174`.
  Exact-copy/zero initialization preserves the post-construction CPU RNG
  SHA-256 `d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
- Active manifest identity:

  ```text
  schema:       25
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_fp32_owner_logs_zero_preserving_variance
  top:          v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_s_owned_k_typed_relevance_four_interval_w_stage_private_p2_physical_value_typed_consequence_plus_two_optional_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_terminal_layer_contracts_lane_local_p3_evidence_mmdit_dense512_execution_gripper_private_codec_closed_event
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_mean_one_transport_codec_closed_event_v120_decay_local_global_clip_source_gradient_probes
  runtime:      cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_codec_closed_event_teacher_isolated_finite_spike_matching_metrics
  ```

  The canonical manifest SHA-256 is
  `c29ca3d120f67880aa4e3577688b0961186e0700d630c244cb67d2a5d88fac28`.

Storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache/checkpoint roots moved.

## Verification and run

The retained local suite now passes 170/170, plus two R2 guards for diagnostic
cadence and decoded-event-to-gate reachability (172/172 in the current
working tree). Tests cover full
forward/backward, G1/G2/G3 ordering and N=49
rematerialization, forbidden G conditions, exact P1 axes/microgrid,
chunked/unchunked P1 output and gradients, Teacher isolation, object/camera
permutations, per-type S perturbation locality, fixed-zero typed values,
typed-owner relabeling equivariance, independent detached DINO target,
observed-cell reconstruction, conditional-K real/null conservation, unique
exported-content reconstruction and its forward/reverse gradient paths,
lossless typed common/residual reconstruction and exact source VJP,
typed-free CoarseAction invariance, the single WorldIntentDock W ingress,
absence of CoarseAction/W raw-typed rereads and the S future-owner fence,
single-call W common ownership, near/far causal write isolation,
zero-preserving appearance/common conditioning, exact camera-axis W geometry,
FP32 PSD covariance, per-camera Teacher null-identity moments, absence of
online status ABI, static/dynamic P1 identity separation, exact three-owner P2
query and reverse VJP, static consequence ownership, raw dynamic transition
and bottom reachability, exact-zero dynamic bottom ingress, current-support
exact-zero P2 routing, covariance-sensitive P2 geometry, spatial interval
preservation, W-owned S selection, no-null per-type physical terminal,
independent semantic/geometry survival and legal W/S/action reverse paths,
typed consequence interaction ownership and one parameter-free fusion,
unique P3 private operands, absence of factual/effect/static-precision aliases,
lane-local optional null decisions, protected no-null reads, and legal P3/bottom
reverse paths, absence and intervention invariance of the removed
layer-contract trajectory branch, retained terminal-adapter reverse paths,
same-camera Teacher geometry, neutral effect, P2 bounds, zero-preserving
variance VJPs, BF16-underflow-resistant producer logs, exact-zero legacy prior
support, all-invalid finite masked terminals, read-only source-gradient hooks,
pre-clip finite-spike attribution, partial-window gradient ownership, current
metric vocabulary, P2 action-band/type/interval diagnostic retention,
diagnostic output/state invariance, evaluation-only P2 value-intervention
locality and posterior invariance, exact matched-noise/finally orchestration,
gripper codec-branch reconstruction, decoded event-boundary locality,
gripper horizon bands and exhaustive
horizon-by-target-event context accounting, a bounded decision-facing console
backed by the unchanged
lossless JSONL cadence, endpoint lifecycle, optimizer ownership, three-stage
gradient logging and checkpoint rejection.
CPU BF16 validates dtype boundaries, not CUDA memory.

R2 additionally guards mean-one target-weighted raw transport with detached
normalized/direction/unweighted-raw audits, exact-copy spatial/terminal P2
query identity and reverse ownership,
typed consequence interaction locality, gripper zero-state identity/locality,
decoded-event-to-gate reachability, arm/auxiliary/motion isolation, production
optimizer ownership and the exact current state-key inventory. The transport
audits reuse rows already computed by the active objective. P2, W and gripper
state/projection scalars run only on the existing diagnostic cadence: every
20th formal training batch and the configured 16 validation batches.
Activation VJP hooks attach only to ephemeral training-forward tensors; named
parameter gradients are read after backward and before clipping, so no hook
accumulates across batches. Execution supervision still computes its
pre-existing candidate tensors on every training batch, but does not retain the
new gripper tensor surface outside that cadence. These additions require no
extra forward, sampling pass or console panel. A fresh local one-batch CPU BF16
forward/backward at the complete production dimensions (`H=512`, 256 patches
per camera, 169,458,596 parameters) completed in 80.79 seconds with finite loss
`3.18261` and finite pre-clip gradient `3.93425`; the retained five-step
deployment guard also passes. The complete source graph is locally closed, but
no R2 GPU smoke or behavioral experiment has run. On 2026-08-28 the user
explicitly authorized the local guard as the temporary substitute because a
separate remote smoke was not available. This does not establish CUDA memory
or throughput: the formal run's startup and first reporting window must enforce
those runtime gates under the following production acceptance:

- finite CUDA BF16 preflight and five-step deployment before the first update;
- batch-eight process peak no greater than 22 GiB;
- aligned batch-2200 early recovery gate against V120;
- all eight epochs and final/mean action, native, first/tail, horizon,
  arm/gripper, event/motion, G/S/W/P and gradient comparisons;
- no late rebound hidden by a best checkpoint.

Use new empty output directories:

```bash
RUN_TAG=schema25_r2_wg01_p202_grip02_smoke
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/smoke_mainline.sh > "${RUN_TAG}.log" 2>&1 &

RUN_TAG=schema25_r2_wg01_p202_grip02_b8
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/train_mainline.sh > "${RUN_TAG}.log" 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema25_r2_wg01_p202_grip02_b8 \
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
