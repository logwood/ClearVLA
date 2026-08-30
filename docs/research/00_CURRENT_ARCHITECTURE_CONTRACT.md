# Current ClearVLA Architecture Contract

Updated: 2026-08-31

This is the compact source of truth for the active independent mainline.
Experiment labels never select model semantics. Historical evidence lives in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`; only still-open work belongs in
`CURRENT_MAINLINE_ISSUES.md`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        28
behavior reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
source reference:       .audit/v120_exact_source_0b92d359/
release status:         Schema28 source/local verification and fresh CUDA BF16 smoke pass; exact-commit batch-eight formal training is active and its first batch-100 audit is finite/ledger-closed; behavior acceptance remains pending
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0, two adjacent learned flows
formal language:        precomputed 4096-wide T5 .pt required; one row selected per sample
bottom:                 V120 seed/transition/CVAE/workspace/Evidence MMDiT/execution
long launcher:          scripts/train_mainline.sh (batch 8, workers 4)
smoke launcher:         scripts/smoke_mainline.sh (batch 1, workers 0)
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only; no optimizer/schedule/RNG load or checkpoint write)
resolved config:        configs/mainline/object_intent_dynamics_323.json
```

> **The executable source is the exact Schema25 replay base plus completed
> R1a/G-01, R1b/G-02, R1c/S-01,S-02, R1d/W-01,W-02, R1e/P1-01,
> R1f/P2-01, R1g/P3-01,B-01, LC-01, R1h/N-01,D-01 and the three
> R2-WG01/P202/GRIP02 structural units, the Schema26 closure and the active
> Schema27 typed-W numerical boundary and the active Schema28 bounded
> action-world closure below.**
> The untouched R0 fingerprint,
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
> fail before their fixed blend. R2 made three source-bounded repairs:
> target-scale-covariant camera-transport supervision, independent exact-copy
> spatial/terminal P2 query owners, and one exact-zero continuous
> gripper-private state for the deployed value/delta heads, followed by a
> codec-decoded absolute/delta event boundary. Event supervision reaches the
> private state through the physically consumed value/delta branches; there is
> no hidden-state event bypass. It added no P3, sampler, checkpoint-selector,
> time schedule, hard event gate or loss-weight bundle. The completed R2
> behavior run then exposed three remaining closure failures:
> inverse-square Teacher-scale weighting reduced responsibility for large/far
> transport, geometry value had no address-level action closure, and the
> categorical gripper head consumed objective budget without entering deployed
> action. Schema26 removes the transport reweighting, adds a zero-preserving
> transport-to-semantic-K address correction inside each physical interval,
> fixes the effect VJP observers at the consumed tensors, removes the
> classifier, and redirects its unchanged `.03` budget to continuous absolute
> and cumulative-delta gripper trajectories from each target event onward.
> It adds no parameter, P3 lane, gain, quota, sampler change or deployment
> event gate. The partial Schema26 formal run then exposed a separate numerical
> boundary: ordinary LayerNorm expanded small typed S residuals to confident W
> directions and allowed a local inverse-standard-deviation gain approaching
> `316`, while public W did not share this amplitude semantics. Schema27 keeps
> every public/generic W operation unchanged and applies the existing
> parameter-free `0.25` variance floor only to typed object/interval/FFN and
> typed W1-to-W2 query/memory normalization. Its local normalization gain is at
> most `4`; exact zero/constant stays zero and small typed amplitude remains
> visible. It adds no parameter, buffer, state key, RNG draw, gain, clip or
> loss. Schema26 and older checkpoints are not exact-resume sources for
> Schema27. Schema28 changes the top/bottom/training/runtime ABIs again; all
> earlier schemas are rejected for exact resume and the next formal run is
> fresh.

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

Schema27 repairs only the typed W numerical chart built on that ownership:

- the public/generic W block and generic W1-to-W2 read retain their inherited
  ordinary LayerNorm modules and values;
- typed common/near/far object attention, interval attention and FFN, plus the
  typed W1-to-W2 query/memory read, use
  `(x-mean(x))/sqrt(var(x)+0.25^2)` in FP32 and return to the caller dtype;
- the floor is read from the existing resolved V120 routing-floor config. It
  adds no affine parameter and cannot synthesize a typed value from zero;
- diagnostic batches record only three aggregate normalization scalars
  (minimum denominator, maximum local gain and maximum output/input RMS ratio)
  plus six type/common-or-interval W-ingress VJPs. The ingress hooks observe a
  W-only identity view, so they do not absorb the same S tensor's legal P2/P3
  gradients;
- under Schema27, W remained cached once per observation. Teacher, losses,
  optimizer ownership, P2 consumers, deployment call count and checkpoint
  tensor inventory were unchanged.

Schema28 closes the first bounded action-world loop without claiming an
environment or fixed-point closure:

- `PhysicalActionCondition` is W's only action ABI. It contains normalized
  physical interval means `[B,4,7]`, the four adjacent deltas anchored at the
  observed current action, and no proposal hidden coordinate. Both initial
  training supervision and deployment refinement use the same deterministic
  projection of a deployable 24-row action prefix: rows `3:8`, `7:16`,
  `15:24`, and `23:24` in zero-based slice notation;
- W is now the goal-invariant map
  `T(ObjectWorldBelief, PhysicalActionCondition)`. It cannot accept goal, an
  S carrier or `CoarseActionIntent.tokens`. Its typed values come from the
  compact current G belief; S remains available only to the later P2/P3 goal
  evaluator;
- `CandidateWorld` stores the W prediction and the exact action-condition
  object atomically. P2 rejects any different condition object. This is a
  Python lineage-identity guard, not a numerical hash and not a promise that
  separately reconstructed equal tensors share identity;
- ordinary deployment and validation run one complete proposal ODE pass,
  rebuild W from that decoded 24-step proposal, then run one complete refined
  ODE pass from the exact same initial physical noise. G, S, cached factual P1
  detail and transition source are built once. Training still materializes W
  once and does not backpropagate through a second deployment sampler;
- the second pass may move away from the action that conditioned its W.
  Validation therefore records final interval/delta mismatch as a residual.
  No zero assertion, convergence claim or fixed-point label is legal;
- the former S-to-W `WorldIntentDock` and goal-attention reader are absent.
  The replacement bias-free physical projection is the only new W parameter.
  Removed initialization draws are consumed and the replacement uses a
  deterministic side stream so retained/downstream fresh initialization and
  the post-construction RNG digest remain unchanged;
- continuous gripper supervision now separates event-row transition from
  between-event persistence. Every event reanchors the absolute branch and
  only strictly later local deltas accumulate; pre-event deltas cannot own a
  later segment. The existing `.03` budget, physical decoder and runtime
  action field remain unchanged;
- execution capacity logits and the near-one schedule interpolation remain
  FP32 through the contraction boundary. This prevents CUDA BF16 from rounding
  a live capacity to exact one; it changes no schedule, controller ownership,
  basis count, loss weight or clipping rule.

Schema28 does **not** unify `ControlledTransition` with W, add persistent object
identity, propagate future null/confidence through P2, execute a robot action,
read a new observation, update belief, reconstruct all 18 flow coordinates or
form a fixed point. Those remain explicit later-stage work.

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
- Schema26 gives geometry one second, non-value responsibility at the spatial
  boundary. For every physical interval it evaluates the same covariance-aware
  coordinate score at `current + transport` and at `current`, subtracts them,
  aggregates legal cameras with the producer-owned conditional camera measure,
  centres over legal K and applies `tanh`. This bounded correction is added
  only to the semantic K logits in that same interval. It does not alter
  geometry K*C selection, interval termination, semantic value amplitude or
  support. Transport zero, absent camera support and K-uniform correction are
  exact-zero identities. No parameter, gain, type gate or time vote is added.

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

ActionIntentDock public S context
    -> causal clean CoarseActionIntent
    -> normalized physical proposal [B,4,7]
    -> PhysicalActionCondition
       absolute interval means + current-anchored adjacent deltas [B,4,14]

compact ObjectWorldBelief from current G + PhysicalActionCondition only
    -> W1: current typed facts once, then 4-8 and 8-16 near innovations
    -> W2: 16-32 and 32-48 far innovations, read-only over completed W1
    -> semantic FutureObjectDynamics [B,4,K,D]
       plus camera transport/covariance [B,4,K,C,2|3], no predicted status
       copied current observable object/camera probability + log probability
    -> CandidateWorld(action-condition identity + dynamics)

deployment/validation bounded outer closure
    -> first complete five-update ODE + endpoint from initial CandidateWorld
    -> decoded proposal [B,24,7]
    -> same deterministic 24-to-4 PhysicalActionCondition projection
    -> rebuild W once from cached ObjectWorldBelief
    -> second complete five-update ODE + endpoint from identical initial noise
    -> final action plus explicit world/action interval and delta residuals

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
    -> covariance-aware (score(current + transport) - score(current))
       -> legal-camera aggregation -> legal-K-centred bounded correction
    -> semantic K selection with that address-only correction and independent
       geometry K*C selection within each I
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
    -> 18-D physical velocity plus the retained motion head
    -> decoded gripper events are evaluation-only behavior of the physical field
```

The history-action proposal remains a supervised auxiliary prediction. Its
future proposal tokens do not enter G/S/W/P, transition or bottom. The
separately compressed executed-action history remains an observable condition
in the shared seed. Generic trajectory/workspace ingress is algebraic neutral;
protected consequence is written once through its no-null ingress. The raw
dynamic P1 residual is read separately without a null and joins the two
independently selected P3 innovations only at the retained optional-ingress
scale. CoarseAction's hidden tokens also do not enter W; only its supervised
physical `[B,4,7]` head does.

Training-only graph:

```text
future DINO supports
    -> FP32/no-grad Teacher association, once per training batch
    -> four semantic and camera-resolved FutureObjectDynamics targets
future action/state + current_loss_support + teacher targets
    -> whole-segment recognizer and auxiliary losses only
first 24 rows of future action
    -> same deterministic 24-to-4 projection used by deployment
    -> CoarseAction physical-head supervision only
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
   fixed-zero null comparison. No S or goal value may enter W. CoarseAction
   has no typed field and only its normalized physical head crosses the W
   action boundary. Factual P1 retains reduced S context; `PolicyIntentDock`
   exposes existing typed common/residual metadata solely to P2, after W owns
   the predicted future values and K/C spatial authority remains explicit.
8. W accepts only `ObjectWorldBelief + PhysicalActionCondition`. W1 owns
   current typed common exactly once and the two near innovations. W2 may read
   completed common/near state but writes only the two far innovations. Every
   interval is one processed current owner plus its matching action-conditioned
   innovation. Goal, S, proposal hidden, Teacher and noisy ODE action are
   structurally absent from the W API. The only W value below W is directly
   supervised `FutureObjectDynamics`; no public or private free W carrier
   crosses into P.
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
    that state. Arm, four auxiliary gripper coordinates and motion retain the
    base action read. There is no final event classifier or event logit in the
    deployed output. The raw target event threshold is used only to start a
    continuous training mask. Each event row owns continuous absolute and
    local-delta transition error; rows strictly after that event own
    persistence until the next event reanchors the segment. No-event samples
    are exact zero for both owners and pre-event deltas cannot leak into a
    later segment. Decoded event precision/recall/F1 remains an evaluation of
    deployed action, never a runtime gate. Capacity logits and the
    near-identity interpolation remain FP32 until the contraction forms its
    update; the capacity remains continuous and non-expansive, not hardware
    rank reduction.
15. Online boundaries use ordinary autograd. No artificial gradient, hard
    gate, entropy/mass quota, scalar progress loss, forced diversity or forced
    nonzero flow is legal.
16. Formal training fails without the configured T5 file. Only explicit
    null-goal smoke may omit it. The established Pen artifact remains a
    one-condition file. A hierarchical multi-task dataset may instead use the
    typed T5-v1.1-XXL instruction bank: the loader binds each episode's exact
    HDF5 instruction to one cache row before device transfer, and every online
    sample still exposes exactly one `[L,4096]` token sequence plus its real
    mask to S. Missing mappings fail before training; this selection adds no
    model parameter, loss, dropout, or optimizer owner.
    The isolated RDT external adapter additionally owns a content-verified
    per-task split manifest with a separate `external_test` lane, ordered
    camera/key identity, and named qpos/action chart profiles. Its right-arm
    profile converts only observed qpos gripper into native command units
    before the action normalizer; state normalization retains qpos units.
    RDT has no inherited Pen event threshold, so its shuffled train loader
    fails closed while that semantic decision is unresolved. These external
    contracts do not claim three-camera or 14-D model consumption.
17. Fresh runs require an empty output directory. Exact resume verifies
   manifest, source/data/language, model/optimizer/scheduler and RNG. Older
   schemas are rejected; explicit compatible bottom-only migration is the
   only migration path.
18. A `CandidateWorld` and its `PhysicalActionCondition` are one atomic lineage
    object. P2 may not consume a retagged or stale W. Normal deployment uses
    exactly one outer W rebuild and identical initial noise across both ODE
    passes. The final action/world mismatch must be finite and logged; it is
    not required to be zero and cannot be called a fixed point.

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

ObjectWorldBelief
  compact current content / typed facts / camera geometry [B,K,(C),*]
  excludes goal, S values, dense chart and training targets

StatelessIntentBundle (serialized compatibility name: ObjectIntentState)
  protected goal/history/public-object tokens
  public / policy interval carriers                      [B,4,H]
  typed common mass / value                              [B,K,3,1|R]
  typed interval-residual mass / value                   [B,4,K,3,1|R]
  typed policy components                                [B,4,3,H]
  temporal queries / state-change evidence               [B,24,H] / [B,H]

Consumer views
  ActionIntentDock (typed-free public action context)
  FactualIntentDock (named reduced factual S context)
  PolicyIntentDock (reduced P3 context plus existing typed P2 metadata)

PhysicalActionCondition
  normalized physical interval action / adjacent delta   [B,4,7] each
  observed current action anchor                          [B,7]

CandidateWorld
  atomic PhysicalActionCondition lineage + FutureObjectDynamics

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

GripperTrajectoryTrainingBoundary
  clean absolute / local-delta / reanchored persistence    [B,24,1] each
  disjoint event-transition / between-event masks          [B,24]
  continuous target                                       [B,24,1]
  consumer: `.03` continuous training loss only; absent at deployment
```

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| G | current DINO/raw history, coordinates, learned flow, current state | T5, action history, proposal, noisy action, Teacher |
| global grounder | completed G3 chart/typed local candidates; detached current DINO and observed mask for its sole reconstruction loss | S, W, noisy action, future Teacher data |
| S | T5, state/executed history, typed ObjectFactSet | frame progress, phase label, noisy action, Teacher |
| W | compact current `ObjectWorldBelief`, normalized `PhysicalActionCondition` absolute/delta | goal, any S value, CoarseAction hidden, target/noisy ODE action, Teacher, predicted status/support, free W value |
| static P1 | completed progressive chart, S, clean action bases | global-K value, W, proposal, Teacher, noisy action/time, second visual read |
| dynamic P1 | action query, Euler time, cached factual base | vision reopen, W, proposal, Teacher, factual relabeling of its live residual |
| P2 | exact three-term query, action-tagged `CandidateWorld`, current chart/camera support; S metadata only as evaluator context after W spatial selection | untagged/stale W, RGB/DINO reopen, predicted status/support, S-owned K/C/time value, type competition, free W hidden |
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
  the sole active measure with ordinary current-camera support masking;
  Teacher magnitude never redistributes row responsibility. Covariance remains
  raw-coordinate and per camera. Detached normalized and direction transport
  audits are computed only on diagnostic batches and never enter backward. The internal
  `0.55/0.15/0.05` semantic/transport/covariance coefficients and outer future
  weight are unchanged. No successor duplicate or status objective remains.
- The former `.03` categorical event budget is renamed
  `gripper_trajectory`. Schema28 keeps that budget but separates two disjoint
  owners: event rows regress clean absolute and local delta transition, while
  later rows regress absolute and event-reanchored cumulative-delta
  persistence until the next event. The threshold selects rows only; it does
  not binarize the target. No event classifier, class boost or focal term
  remains. All other action, future, flow geometry, intent scaffold, history
  proposal and execution-value external weights are unchanged from the
  recovery reference.
- The whole-segment recognizer supervises only S's public interval carrier.
  S typed relevance is trained through factual/P2/P3/final action paths and no
  longer supplies W values. The typed-free coarse-action physical head is
  supervised from the first 24 future rows through the exact deployment
  24-to-4 projection; its hidden tokens are not a W input. No public future
  target, entropy or usage loss directly trains S relevance.
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

- Observation/G/S, exact static P1 and transition source build once per
  observation. Initial W builds once from the clean physical proposal;
  deployment/validation rebuild only W once from the first decoded proposal.
- Each of the proposal and refined passes runs dynamic P1/P2/P3, transition,
  layer contracts and bottom at action-update times `[0,.2,.4,.6,.8]`, then
  once at `1.0` for the retained motion head. Both passes use identical initial
  physical noise. The endpoint call cannot change the integrated action.
  Decoded gripper events come directly from the second integrated action.
- The ordinary training loss forward materializes W once and runs no outer
  deployment sampler. A diagnostic validation batch records refinement
  pre/post/action/world changes plus the nonzero-allowed final mismatch.
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
  completed R2 model has `169,458,596` total / `153,087,865` trainable
  parameters, 1,388 parameter tensors, 1,066 trainable/optimizer tensors, 23
  optimizer groups and 1,394 state-key names. Its ordered state-key-name
  SHA-256 is
  `384bf6aa4f765382f3d7b4251f0b70f53fe233d3a86090f8ea2bdad6d886d174`.
  Exact-copy/zero initialization preserves the post-construction CPU RNG
  SHA-256 `d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
  Schema26 removes that final `2 -> 3` classifier (`-9` trainable parameters,
  `-2` parameter tensors and `-2` state keys). Raw transport supervision and
  the P2 address correction are parameter-free. The active model therefore has
  `169,458,587` total / `153,087,856` trainable parameters, 1,386 parameter
  tensors, 1,064 trainable/optimizer tensors, 23 optimizer groups and 1,392
  state-key names. Its ordered state-key-name SHA-256 is
  `eb9b6077e51f9ed6ec65f3462e34e061034913fbfa9d19b745599d4b34afc88d`.
  Removing the exact-zero classifier consumes no RNG, so the construction RNG
  digest remains unchanged. Schema27 adds only parameter-free normalization
  modules and ephemeral scalar/VJP observers. It therefore retains all
  Schema26 parameter counts, optimizer groups, state-key names, the ordered
  state-key digest and the construction RNG digest exactly.
  Schema28 removes the bias-free H-wide W goal attention (`in_proj_weight` and
  `out_proj.weight`, `-1,048,576` parameters at H=512) and adds one bias-free
  `14 -> H` physical action-condition projection (`+7,168`). The exact net is
  `-1,041,408` trainable parameters and `-1` parameter/state tensor. Gripper
  reanchoring and FP32 capacity execution are parameter-free. The active model
  has `168,417,179` total / `152,046,448` trainable parameters, 1,385 parameter
  tensors, 1,063 trainable/optimizer tensors, 23 optimizer groups and 1,391
  state-key names. The ordered state-key-name SHA-256 is
  `70a8a5be21de40c460de6cff899942d5331837700db289350a0b1920c133b053`.
  Initialization-only retirement plus the deterministic replacement side
  stream preserves the post-construction CPU RNG SHA-256
  `d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
- Active manifest identity:

  ```text
  schema:       28
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_fp32_owner_logs_zero_preserving_variance
  top:          v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_s_owned_relevance_goal_invariant_physical_action_conditioned_w_single_consequence_refinement_p2_transport_address_typed_consequence_two_optional_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_terminal_layer_contracts_lane_local_p3_evidence_mmdit_dense512_execution_fp32_capacity_gripper_private_continuous_field_no_event_head
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_raw_transport_event_transition_persistence_gripper_trajectory_v120_decay_local_global_clip_physical_w_ingress_gradient_probes
  runtime:      cached_observation_progressive_gsw_exact_p1_physical_action_tagged_w_single_refinement_v120_nodes_clean_endpoint_decoded_gripper_events_teacher_isolated_finite_spike_matched_p2_value_address_capacity_metrics
  ```

  The canonical manifest SHA-256 is
  `d26bf48ea1391f691662640c63ba17a23e56fd50cff7fab433d36e0af4930528`.

Storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache/checkpoint roots moved.

## Verification and run

Historical behavior evidence is still Schema26: three complete validations
plus epoch-4 batch 360, median `1.809 s/batch`, `12.020 GiB` process peak and
64 finite spike events. Schema27 passed its local source suite and one
production-dimension CPU BF16 batch, but never produced CUDA behavior evidence.
Neither result proves Schema28.

Schema28's complete relevant mainline/runtime/auditor selection passes
`247/247`, with touched-file Ruff, py_compile and diff checks also passing.
They add explicit checks for:

- identical training/runtime 24-to-4 physical action projection;
- W invariance to goal, S typed values and coarse hidden coordinates;
- gradients through current G facts and the physical action head only;
- atomic CandidateWorld action lineage and stale-world rejection at P2;
- exactly one outer W rebuild, identical initial noise and finite pre/post
  semantic/transport/action diagnostics;
- finite final action/world mismatch without asserting a fixed point;
- event-local gripper transition, reanchored persistence and no pre-event
  delta leakage;
- FP32 near-one capacity and retained contraction VJP;
- matched proposal, P2 and execution validation counterfactual schedules;
- current manifest/state/optimizer/RNG inventories and checkpoint rejection.

One fresh production-dimension CPU BF16 batch (`H=512`, 256 patches per
camera) completed in `95.829 s` with finite loss `2.51860929` and finite raw
pre-clip gradient `3.20619488`. The bounded deployment probe completed in
`20.383 s`, observed exactly 12 dynamic calls (two copies of five updates plus
their endpoint heads), two W materializations and one outer W refinement, and
kept action, physical field and final interval/delta mismatch finite. Capacity
was exactly one at this first warmup step by schedule; the separate near-one
FP32 contraction/VJP test covers the post-warmup numerical boundary.

CPU BF16 validates dtype and backward boundaries, not CUDA memory or behavior.
The fresh CUDA BF16 smoke
`schema28_action_world_smoke_20260831_012955_r1` then completed at commit
`097330a894d948d66c419f8af07325a5b0ff712e`: median runtime was
`1.549 s/batch`, measured process peak was `3.974 GiB`, both complete ODE
passes executed with exactly one W refinement, all closure fields were finite,
and proposal-to-refined action delta was `3.75218e-05`. Final interval/delta
mismatch was finite and nonzero (`3.42142e-05 / 1.84383e-05`), as allowed by
the non-fixed-point contract. The two finite raw-gradient spikes (`6.17` and
`7.56`) were both owned by the randomly initialized arm output head.

Fresh batch-eight formal training started from the same exact commit as
`schema28_action_world_b8_20260831_013140`. Its first archived audit through
batch 100 has exact-zero group-ledger error, contribution gap at floating-point
noise (`5.96046e-09` in the first full window), finite W/P2/action gradients,
no forbidden goal/coarse-hidden W ingress, no action-lineage error, median
runtime `1.838 s/batch`, and a `9.1%` decrease in total loss across the five
available windows. Three threshold crossings through batch 100 were again owned
by `bottom.decoder.velocity_head.arm_abs.weight`; no non-finite or new
normalization-owner spike was observed. This is startup health evidence, not
trained behavior evidence.

Production acceptance remains:

- continued finite closure residuals, capacity, action, W/P2 and gradient
  metrics after warmup and across every completed epoch;
- batch-eight process peak no greater than 22 GiB;
- aligned batch-2200 early comparison against V120, R2, Schema26 and Schema27;
- all eight epochs and final/mean action, native, first/tail, horizon,
  arm/gripper, event/motion, G/S/W/P, capacity and gradient comparisons;
- no late rebound hidden by a best checkpoint.

Use new empty output directories:

```bash
RUN_TAG=schema28_action_world_smoke_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/smoke_mainline.sh > "${RUN_TAG}.log" 2>&1 &

RUN_TAG=schema28_action_world_b8_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/train_mainline.sh > "${RUN_TAG}.log" 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema28_action_world_b8_TIMESTAMP \
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
