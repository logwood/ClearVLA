# Current ClearVLA Architecture Contract

Updated: 2026-09-01

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
release status:         Schema28 remains the completed behavior reference; the d8a77a1 source-local cache-isolation repair passed the real Pen B8 CUDA v2 VJP gate and fresh Pen/RDT-8 smokes; two fresh Schema29 formal runs are active, but no completed Schema29 behavior curve is valid yet
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
> Schema27 typed-W numerical boundary, the Schema28 bounded action-world
> closure and the active Schema29 train/runtime alignment below.**
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
> earlier schemas are rejected for exact resume. Its completed formal run is
> fresh and behavior-audited below. Schema29 retains its parameters and
> deployment graph but changes the training/runtime ABI again; Schema28 is not
> an exact optimizer-resume source. The first Schema29 CUDA launches are not
> behavior evidence: a later real-batch parameter VJP proved that pass0 had
> polluted the enclosing BF16 autocast weight cache and severed formal-pass
> parameter edges. The repaired lifecycle below must pass a fresh CUDA gate
> before either outlet restarts.

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
  detail and transition source are built once. Schema28 training materialized
  W once and did not backpropagate through a second deployment sampler;
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

Schema29 aligns the formal training action with the deployed action-conditioned
world without adding a sampler pass or a second model:

- observation/G/S/static P1/transition source and Teacher still build once;
- one `FlowMatchingState` is sampled once. A cache0 velocity call under
  `no_grad` estimates its clean endpoint, the deterministic codec decodes it,
  and `PhysicalActionCondition.from_horizon_action` creates a fully detached
  condition;
- only W is rebuilt from that condition. A second velocity call on cache1 uses
  the exact same noisy physical field and flow time and alone owns the existing
  action/event/motion/execution losses; the existing future objective also
  consumes cache1 dynamics;
- pass0 runs inside a forked CPU/current-CUDA RNG scope. Pass0 and pass1 begin
  from the same dropout state, while the final global RNG state advances by
  exactly one formal dynamic call as in Schema28;
- the enclosing formal autocast cache remains enabled, but every parameterized
  no-grad call that can share its lifetime with a later attached call uses a
  nested autocast context with `cache_enabled=False`. This applies to pass0,
  the native candidate target probe, and the retired-sequential
  learned-execution hard audit. The later formal block/head calls retain the
  normal cache. Disabling the cache globally, changing compute dtype, or
  detaching the formal activations is not allowed;
- pass0 has no loss and no backward path. The condition input VJP is exact zero
  by design; the post-projection, post-`0.35` W action carrier is the named
  consumer-gradient observation point;
- parameters, buffers, state keys, optimizer groups, objective weights,
  clipping, five-step deployment sampler and two-pass deployment call count are
  unchanged. This is one detached training approximation, not a fixed point.

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

training detached action self-conditioning
    -> sample one FlowMatchingState
    -> pass0 velocity(cache0) under no-grad and forked RNG
    -> decoded clean endpoint -> detached PhysicalActionCondition
    -> rebuild W only -> cache1
    -> pass1 velocity(cache1) from the same noisy field/time
    -> compose the existing losses once; future loss reads cache1 dynamics

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
    camera/key identity, and named qpos/action chart profiles. The manifest
    canonicalizes root-relative POSIX episode identity independently of
    machine-local `pathlib` component discovery order, then maps accepted
    identities back to the caller's loaded-episode indices. It covers the
    complete source inventory while recording any episode shorter
    than the fixed 73-row `-24/+48` typed window as an explicit identity/length
    exclusion; only eligible episodes enter the four disjoint split lanes.
    Its right-arm profile converts only observed qpos gripper into native
    command units before the action normalizer; state normalization retains
    qpos units. Manifest-backed RDT loading requires a per-instruction T5 bank
    whose complete source episode count and instruction multiset digest match
    the live inventory; the legacy one-row condition remains legal only for
    the Pen path. The released RDT corpus already supplies task-local
    `lang_embed_0.pt` rows: the official encoder script defines index zero as
    the original instruction, before its simplified and expanded variants.
    The adapter therefore builds the typed bank from those existing BF16
    `[L,4096]` rows without loading or downloading T5. Repeated exact texts are
    never averaged: every candidate path and storage digest is retained, and
    the adopted lexicographic policy selects one stable root-relative source.
    The generic encoder is only an explicit, local-only fallback; network
    download additionally requires a separate acknowledgement. A loader-only
    smoke may materialize a bounded deterministic
    split subset after verifying the complete source, manifest, train
    normalizer and language identities, while formal loading still requires
    DINO cache rows for every model-materialized episode. The data bundle owns
    that materialized episode union once and supplies the same scope to the
    token reader and dataset/checkpoint identity. The complete raw inventory,
    split manifest, train normalizer and language bank remain source-wide;
    unselected tasks and the preserved `external_test` lane cannot become
    model-cache requirements through the identity path. RDT has no inherited Pen event
    threshold. Its continuous transition owner is the adjacent command stream:
    the first policy row compares with the preceding executed command and later
    rows compare with the preceding policy command. Observed qpos remains the
    codec/physical-decode anchor and is never converted into a discrete event
    label. The eight-task shuffled train loader uses only the explicitly
    adopted train-only adjacent-command p95 activity threshold
    `0.18310546875` raw command units.
    These external contracts do not claim
    three-camera or 14-D model consumption.
    The real bounded `val` acceptance at commit
    `9a5611ede2133a5365d02e3a73b1a1fe5a6eb841` closed this external boundary:
    6,131 source / 6,120 eligible episodes and 11 explicit short exclusions,
    canonical manifest SHA-256
    `2442ecd9c382d14123449a5b72d408bad4bcf84b164f6104a0d615cf5925212b`,
    a corpus-derived 271-row language bank, one three-camera DINO cache, and
    one finite two-camera/right-arm 7-D typed batch. The report explicitly
    records `model_constructed=false` and `optimizer_constructed=false`; it is
    not model-side multiview, bimanual, depth, backward, checkpoint or
    experiment evidence.
    The adopted first-round bounded RDT scope is now a content-verified exact
    eight-task selection layered on that same manifest. A task-complete CPU
    audit covers all 302 `rdt_data` directories and all of each task's
    episodes; selection then retains 179 eligible episodes in deterministic
    task order (`143/18/18` train/val/test), 81,237 frames and 68,349 fixed
    typed windows (`54,648/6,711/6,990`). Semantics come only from the scalar
    HDF5 instruction. Every selected episode is finite native 14-D, has
    complete decodable high/right-wrist RGB, and is projected to native source
    coordinates `7..13` for the unchanged right-arm 7-D model ABI. The
    all-episode left-role audit combines numeric action/qpos/gripper evidence
    with five uniformly spaced high-camera frames per episode and records that
    no selected task requires left-arm support or collaboration.
    The exact selection SHA-256 is
    `99f082028be7d9b92f0be4ed02ce22b5367f5f4c7274bbb1064b403770d1fd6f`.
    One shared train-only normalizer covers all 143 selected train episodes
    and 64,944 rows; per-task normalizers are forbidden. Its canonical
    SHA-256 is
    `1aa44936eb3fa659270a2dcc2a0258fa1e888332a37de5034a56ccef09320e0c`.
    The reusable DINO cache was written directly from HDF5 RGB in fixed
    `(high,left_wrist,right_wrist)` order, while the first-round model reader
    selects only `(high,right_wrist)`. Its 179 FP16 arrays occupy exactly
    95,831,087,488 bytes before episode/report metadata; the complete token
    inventory SHA-256 is
    `2d9379804effa65968e4e8b19b032acd0b3e353f6dd3c468b774b9e26ee1833d`.
    The 26 source `/test/` episodes remain `external_test`, are not
    materialized by this bounded formal loader and cannot enter training or
    tuning. Task ID remains CPU audit/sampling/logging metadata and is absent
    from model samples. The train-only adjacent-command audit adopts p95
    `0.18310546875` raw command units for all eight tasks: total activity-window
    fraction is `0.269379`, and the lowest nonzero task coverage remains
    `0.123409`; `press_stapler=0` is a constant-command data fact and receives
    no fabricated event. P97.5 would reduce `draw_triangle` to `0.03646`, while
    p90 would broaden total activity to `0.326618`. This preparation therefore
    still does not claim
    depth, a three-camera model consumer, or native 14-D bimanual adaptation.
    The adopted eight-task experiment interface is now source-complete but
    behavior-pending. It constructs one immutable CPU registry from the
    selection manifest (`episode_idx -> task index`) and uses it in exactly
    three non-model places. Shuffled training assigns batch slots to tasks
    before applying the existing uniform/event/motion information lanes; at
    batch eight, every batch contains exactly one row from every selected task,
    and an absent informative pool falls back only to that task's uniform
    pool. A bounded validation budget selects an equal deterministic panel per
    task instead of truncating the task-sorted dataset. One shared deployment
    prediction is then sliced for per-task full/three-band/arm/gripper/event
    accounting, plus sample-weighted micro and equal-task macro summaries.
    Missing tasks are listed through coverage and never receive fabricated
    zero performance. Actual train sample counts/fractions, the task registry,
    sampler and validation-panel summaries are serialized as run/data state;
    task metadata owns no gradient, module, optimizer tensor or checkpoint
    tensor and never enters `TrainingBatch.online`.
    `scripts/train_rdt_multitask.sh` and `scripts/smoke_rdt_multitask.sh` are
    the only RDT model launchers. The eight-task config and both launchers pin
    `0.18310546875`, atomically binding sampler event rows, continuous
    gripper-trajectory ownership and decoded validation; the formal launcher
    rejects environment or CLI threshold overrides. Pen remains exactly
    `0.10`; unlike thresholds fail configuration validation. The former candidates
    `0.146484375 / 0.40283203125 / 0.7599645256996155` are invalid because their
    first row mixed command with qpos. The v2 audit uses adjacent commands only;
    qpos is retained only as a physical-boundary audit. Sampler strata, the
    continuous trajectory mask and decoded event metrics consume the same
    adopted command-activity threshold, while the physical action field keeps
    its qpos-anchored first delta. The first mixed-model backward/deployment
    smoke reached this interface, but its training-validity claim was later
    withdrawn after the shared CUDA BF16 weight-cache defect was found. It must
    be rerun from fresh state after the real-batch parameter-VJP gate passes.

    The active gripper boundary map is therefore:

    ```text
    HDF5 command[t-1], command[t]
      -> profile projection / train-only normalization
      -> ActionSupervision.gripper_transition_boundary [B,7] FP32
      -> sampler activity stratum
      -> loss event/transition/persistence masks
      -> decoded validation event class

    HDF5 qpos[t] -> command-chart projection
      -> ObservableHistory.action_state [B,7] FP32
      -> codec encode/decode and first physical-field delta
      -> formal flow, decoded action and physical-delta consistency

    .03 continuous gripper objective
      -> event-owned absolute + codec-delta/persistence field
      -> deployed gripper-private value/delta channels
      -> bottom-head gradients
    ```

    The previous-command tensor is detached dataset supervision, not an online
    model input. Its non-gripper coordinates have no consumer. It adds no model
    parameter, optimizer owner or checkpoint tensor; the declared boundary is
    serialized in data/run identity. The adopted p95 threshold remains identical
    across all three mask/metric consumers and is serialized with the data/run
    identity. Using command delta as the codec's first delta target is
    forbidden because it would compete with the qpos-anchored formal flow target.
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
- The ordinary training loss forward samples flow once, executes one detached
  cache0 endpoint estimator, rebuilds only W, then executes one formal cache1
  dynamic pass and composes the loss once. It runs no outer deployment sampler.
  A diagnostic batch records pass0/pass1 action, W-change and final
  world/action mismatch scalars; full tensors are not logged.
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
  Schema29 adds only ephemeral forward tensors, scalar diagnostics and one
  activation-gradient hook. It retains all Schema28 parameter/state counts,
  optimizer groups, ordered state-key digest and construction RNG digest.
- Active manifest identity:

  ```text
  schema:       29
  observation:  restored_v120_three_frame_flow_dino_progressive_g123_fp32_owner_logs_zero_preserving_variance
  top:          v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_s_owned_relevance_goal_invariant_physical_action_conditioned_w_single_consequence_refinement_p2_transport_address_typed_consequence_two_optional_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_terminal_layer_contracts_lane_local_p3_evidence_mmdit_dense512_execution_fp32_capacity_gripper_private_continuous_field_no_event_head
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_raw_transport_event_transition_persistence_gripper_trajectory_v120_decay_local_global_clip_detached_endpoint_self_conditioned_w_single_action_loss_rng_matched_gradient_probes
  runtime:      cached_observation_progressive_gsw_exact_p1_physical_action_tagged_w_train_cache0_endpoint_cache1_formal_pass_deploy_single_refinement_v120_nodes_clean_endpoint_decoded_gripper_events_teacher_isolated_finite_spike_matched_p2_value_address_capacity_metrics
  ```

  The canonical manifest SHA-256 is
  `96883e89ea3df8e5da1693022bdfff79d92fd3100a1deb55360d608cc897f8e6`.

Storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache/checkpoint roots moved.

## Verification and run

The authoritative Schema28 behavior artifact is
`new_logs/schema28_final_20260831_013140/{run_context.json,metrics.jsonl}` at
commit `097330a894d948d66c419f8af07325a5b0ff712e`. It contains all eight
epochs and 1,144 train windows under seed 0, batch 8, the fixed 63/5 split and
action-normalizer fingerprint `32a3a4d7f21f`. The loss ledger is exact; there
are no tracebacks or non-finite rows. Median throughput is `1.840 s/batch` and
the process peak estimate is `12.112 GiB`.

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

The completed behavior curve reaches its best aggregate point at epoch 6 and
ends at physical full/arm/gripper RMSE
`0.07657 / 0.05677 / 0.14733`. Final bands are
`0.02502 / 0.05513 / 0.09743`; tail/first is `7.658`. Decoded gripper event
precision/recall/F1 are `0.6006 / 0.2749 / 0.3771`, with `621/1357` predicted
versus target events. Schema28 is directionally better than the aligned
Schema26 final point but does not close far-horizon or gripper behavior.

The outer pass changes the action by RMS `0.02514`; its final interval/delta
mismatch is `0.02933 / 0.01514`. Rebuilding W changes semantic/transport by
`0.06639 / 0.00713`. This establishes one useful bounded correction, not a
fixed point. W2 semantic and transport are respectively about `0.69x` and
`0.44x` their Teacher magnitudes. Matched P2 interventions establish strong
far semantic action responsibility and weak learned-scale geometry action
responsibility, but do not separate W consequence from ControlledTransition.

There are 12 finite threshold crossings in the full run: four observation
flow-delta, four observation target-DINO-key, three arm-output-head and one
raw-flow-pyramid owner. The maximum global preclip is `24.63`. The Schema27
typed-normalization owner does not recur, while older observation owners remain
an open empirical boundary.

At the completion of the Schema28 behavior run, it was source-closed and
runnable but not behavior-closed. The next architectural unit was deliberately
held outside the contract until matched W/consequence/ControlledTransition
attribution and the complete producer/consumer review selected it. The
subsequent evidence below selected the Schema29 train/runtime alignment while
still rejecting an extra W-to-transition bridge and a transition-owned world
generator.

The post-run Stage-A attribution source adds only non-persistent evaluation
state and scalar accounting. It reuses the refined cache and identical initial
noise to compare explicit-none, W-dynamic neutral, consequence-effect neutral,
CT-delta neutral, joint W+CT neutral and deterministic wrong-action W. It adds
no parameter, buffer, state key, optimizer owner, RNG draw, loss or backward.
The current inventory remains exactly `168,417,179 / 152,046,448`, 1,391 state
keys and 23 optimizer groups. The relevant mainline/runtime/checkpoint/auditor
selection passes `223/223`, and one complete fresh CPU FP32 validation batch
passes all intervention identities and cleanup checks. These local results
validate implementation algebra. The authoritative Schema28 checkpoint replay
has now completed on all 179 validation batches with a 16-batch diagnostic
subset. Primary/explicit-none and W-neutral/consequence-neutral are bit-exact;
all `128/128` wrong-action donor rows are valid and the subset contains 117
target gripper events. W neutral changes far action/gripper by
`0.05829 / 0.13240` and worsens their paired MSE by `0.00557 / 0.03203`; CT
neutral independently changes them by `0.01605 / 0.04142` and worsens MSE by
`0.00094 / 0.00655`. Wrong-action W changes far action/gripper by
`0.00943 / 0.01952`. This selects train/runtime action-conditioned-W alignment
for the next gate while retaining both W and CT; it does not authorize a
W-to-transition bridge, transition-owned world, geometry gain, transport quota
or hard event gate.

The validation-only estimator gate completed on all 179 validation batches
with 16 diagnostic batches. Relative to coarse, its distance to the complete
five-step proposal is `0.210828x` for interval action, `0.115498x` for interval
delta, `0.168057x` for semantic W and `0.221124x` for transport W. Its update
direction cosine is `0.984706` with `1.0` valid coverage. The extra path costs
`0.200390 s/diagnostic batch` and `0.013094 GiB` live allocation. This passed
the predeclared Schema29 gate; it did not itself change training.

The original Schema29 source/local selection proved the call count, detached
condition, loss owner, RNG restoration, parameter inventory and checkpoint
ABI, but it did **not** prove that the formal CUDA BF16 pass retained parameter
edges. That omission invalidates its former CUDA-interface closure claim.

The decisive real Pen batch report is
`new_logs/schema29_real_batch_probe_a671640.json`, schema
`clearvla-schema29-real-batch-gradient-ab-v1`, from clean source
`a6716405ff363657e0acfb9fc6e8b2802accf255`. It ran cache0 as a single
attached pass and cache1 with the actual no-grad pass0 followed by the formal
pass inside one enclosing BF16 autocast context. The total-loss VJPs were:

```text
                                         cache0 attached       cache1 formal
velocity-output parameter L2             3.1299548             0
gripper-gate parameter L2                 0.01231698            0
motion-head parameter L2                  0.05398325            0
physical-velocity activation RMS          7.79045e-4            7.79045e-4
velocity-head-input activation RMS        3.24186e-6            3.24186e-6
```

Forward boundaries remained numerically aligned while activation VJPs stayed
nonzero. The zero appeared only at parameter ownership. This isolates the
cause to PyTorch autocast's cached BF16 parameter copies: pass0 was the first
dynamic parameter call under `no_grad`, its cached casts had no parameter edge,
and pass1 reused them. It is not evidence for a detached W condition, a weak
loss, clipping, optimizer ownership or a zero activation gradient.

The source-local repair leaves the Schema29 graph, parameters, buffers, state
keys, optimizer groups, objective weights, RNG schedule and deployment calls
unchanged. It disables autocast weight caching only inside the three
parameterized no-grad scopes named above and expands the real-batch probe to
every optimizer role, Evidence-MMDiT blocks `0/1/2`, the three output owners,
formal activation VJPs, dtype/detach state and nested cache state. The gate is
relational: every role with cache0 signal must retain it in cache1; a role that
is zero in both modes because of a legal zero initialization is not required to
become nonzero. The relevant CPU selection passes `166/166` with two CUDA-only
regressions skipped; Ruff and py_compile pass. CPU results prove placement and
graph semantics, not the CUDA repair.

The former Pen/RDT smokes at `4125a3d` and the formal runs that reached
`2160 / 1720` steps are invalid as training/behavior evidence. Their finite
losses, exact ledgers, throughput and task coverage remain useful only as
launcher/data-ABI observations. Both formal runs were stopped and must never
be resumed from their checkpoints.

The fresh v2 Pen B8 CUDA BF16 gate has now passed at source commit
`d8a77a19cfbd7520ae790b3938e2d1fb3a8a7a6f`; its local artifact is
`new_logs/schema29_real_batch_probe_d8a77a1_r1.json`. The report's
`worktree_dirty=true` comes only from untracked report/log artifacts in the
dedicated probe worktree; tracked executable source is exact `d8a77a1`.
Pass0/decoded condition are detached, pass0 uses cache-off, cache0/cache1 formal
use cache-on, pass0 velocity is BF16, and CPU/current-CUDA fork RNG is restored.
The decisive total-loss parameter VJPs are:

```text
                                         cache0 attached       cache1 formal
all trainable parameter L2               3.1506655             3.1506667
velocity-output parameter L2             3.1299548             3.1299548
gripper-gate parameter L2                 0.01231698            0.01231698
motion-head parameter L2                  0.05398325            0.05398325
Evidence-MMDiT block 0 L2                 0.0244343             0.0244343
Evidence-MMDiT block 1 L2                 0.0245463             0.0245463
Evidence-MMDiT block 2 L2                 0.0243540             0.0243540
```

Every optimizer role with cache0 signal is preserved in cache1 at approximately
`1.0x`. `bottom_capacity`, `consequence` and `p2_effect_reader` are zero in both
modes at the fresh step-zero boundary; that is the existing capacity warmup and
zero-initialized W/P2 consequence semantics, not cache1-specific attenuation.
The two CUDA-only internal regressions for the candidate target probe and
sequential learned-execution hard audit also pass on the target server.

Fresh B8 smokes then completed from the same commit and source digest
`5294ee5521705adbcc3b85da82755063befd3de68fa5a79bce5b165b01be47a9`:

- `schema29_cachefix_pen_b8_smoke_d8a77a1_r1` completed one optimizer step,
  deploy-style validation, exact ledger, and atomic `latest.pt`/`best.pt`
  writes. Raw global/MMDiT/head gradients were
  `3.02863 / 0.04150 / 3.00206`.
- `schema29_cachefix_rdt8_smoke_d8a77a1_r1` completed the same lifecycle with
  raw global/MMDiT/head gradients `3.89867 / 0.05054 / 3.87384`; its bounded
  validation panel and training mix both cover all eight tasks exactly once.
- Both serialize manifest digest
  `96883e89ea3df8e5da1693022bdfff79d92fd3100a1deb55360d608cc897f8e6`,
  identical module inventory and optimizer ownership. Their one-batch RMSE,
  event F1 and cold-start runtime are interface evidence only, not behavior or
  throughput conclusions.

The fresh formal runs
`schema29_cachefix_pen_b8_d8a77a1_20260901_r1` and
`schema29_cachefix_rdt8_b8_d8a77a1_20260901_r1` were then launched on separate
GPUs with no resume or migration argument. Pen has reached at least step `400`
with an exact ledger, no lineage/non-finite error and all named optimizer roles
carrying a formal parameter gradient. At step `400`, raw global/P2/consequence/
CT/MMDiT/head L2 are `1.959996 / 0.066696 / 2.716e-4 / 0.027799 /
0.306001 / 1.584174`; the step-zero-legal capacity owner has also opened
naturally to `0.001073`. Three finite `arm_abs.weight` cold-start crossings
occurred at steps `8/10/40` (`5.44/5.40/5.36`) and did not recur through step
`400`. After the overlapping cold start, its 20-batch windows are in the
`2.11-2.75 s/batch` range; no release throughput value is selected before a
controlled epoch comparison.

RDT-8 completed its source-wide HDF5 cold start, the same model preflight and
its first 20-batch window. The ledger is exact; raw global/P2/consequence/CT/
MMDiT/head L2 are `2.88159 / 1.627e-4 / 4.102e-7 / 0.004355 / 0.039080 /
2.86530`, with no spike, lineage or non-finite row. Its serialized
`clearvla-task-balanced-information-sampler-v1` contract uses task count `8`
and batch size `8`, so every formal batch contains one row per selected task.
The first-window `6.81283 s/batch` still includes data/model cold start and is
not a steady throughput claim. No completed Schema29 epoch or behavior claim
exists yet. The code change does not raise the manifest schema because it
changes no serialized graph state, but its new source digest intentionally
makes every old checkpoint fail exact resume.

Use new empty output directories:

```bash
RUN_TAG=schema29_self_condition_smoke_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/smoke_mainline.sh > "${RUN_TAG}.log" 2>&1 &

RUN_TAG=schema29_self_condition_b8_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR="runs/${RUN_TAG}" \
nohup bash scripts/train_mainline.sh > "${RUN_TAG}.log" 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema29_self_condition_b8_TIMESTAMP \
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
