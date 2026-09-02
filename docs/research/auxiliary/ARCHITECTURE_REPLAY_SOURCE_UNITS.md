# Cross-version replay source units

Status: source-derived frozen R1 register; G-01, G-02, S-01, S-02, W-01,
W-02, P1-01, P2-01, P3-01, B-01, N-01 and D-01 implemented and statically
closed; S-03 guard verified; T-01 deferred. All eight R1 slices are complete;
the first formal R1 run later completed through epoch 8.
Established on 2026-08-26.

This file records the implementation semantic units recovered across
Schema26-39. It is deliberately source-first:

- ARCHITECTURE_REPLAY_LEDGER.md supplies donor coordinates and historical
  questions only.
- Exact source snapshots and diffs define every unit.
- A log may later help rank a closed candidate, but it cannot define a
  producer, owner, retained axis, consumer, zero semantic or gradient path.
- ARCHITECTURE_REPLAY_PLAN.md remains the sole authority for dispositions,
  bundle selection and implementation order.

The register is organized by live semantic boundary rather than by historical
version. A donor commit can contribute to several units, and one unit can use
source material from several donor commits.

## 1. Source acquisition rule

For each unit, the review coordinate is:

    exact donor hunk
      -> producer and observable inputs
      -> retained axes and representation
      -> every online and training consumer
      -> ordinary action and auxiliary gradients
      -> runtime repetition and checkpoint ABI
      -> least restrictive replay form

The Schema25 source at 6a6c1bf is the comparison base. Later source is not
assumed correct merely because a class, test or comment names an ownership
contract. The then-current Schema39 donor snapshot was used only to inspect
which boundaries survived and how their consumers were wired; it is not the
active checkout named by this historical register.

## 2. Unit index

| Unit | Boundary | Principal donor source | Replay disposition | R1 |
|---|---|---|---|---|
| G-01 | Exact completed G3 transition source | Schema26 policy.py and transition.py | KEEP | yes |
| G-02 | Conditional-K G3 and exported-content reconstruction | Schema27 and Schema30 grounding.py | REIMPLEMENT | yes |
| S-01 | One typed owner and no CoarseAction typed re-entry | Schema26 intent.py and types.py | KEEP purpose; REIMPLEMENT ABI | yes |
| S-02 | Lossless common plus interval-residual typed views | Schema31 and Schema37 intent.py and types.py | REIMPLEMENT | yes |
| S-03 | Future-owner supervision fence | Schema27 versus Schema32 intent.py and top.py | KEEP Schema25 boundary; REJECT direct future head | guard |
| W-01 | One-way common-to-interval W ownership | Schema32 dynamics.py; Schema34 anti-donor | REIMPLEMENT | yes |
| W-02 | Typed W field and online ABI cleanup | Schema27, Schema35 and Schema37 dynamics.py and types.py | REIMPLEMENT | yes |
| P1-01 | Static fact versus dynamic policy residual | Schema35, Schema38 and Schema39 | REIMPLEMENT | yes |
| P2-01 | Spatial selection followed by physical interval terminal | Schema33, Schema38 and Schema39 compiler/effect_terminal.py | REIMPLEMENT | yes |
| P3-01 | Unique P3 values around protected carriers | Schema27, Schema28, Schema37 and Schema39 compiler.py | REIMPLEMENT | yes |
| B-01 | Lane-local bottom reads and protected no-null reads | Schema37 and Schema39 time_domain_mmdit.py | REIMPLEMENT | yes |
| N-01 | FP32 support and finite all-invalid numerics | Schema38 and Schema39 grounding/effect_terminal.py | KEEP mechanics where required | yes |
| D-01 | Source-owner diagnostics and spike attribution | Schema34, Schema38 and Schema39 | KEEP only matching diagnostics | yes |
| T-01 | Teacher association backend | Schema31 teacher.py | DEFER; retain Schema25 row softmax | no |

R1 means the unit was selected for the reconstructed R1 candidate. The label
does not authorize a new training run or a whole-commit cherry-pick.

## 3. Selected source units

### G-01: exact completed G3 transition source

Source anchors:

- caa7e33:clearvla/mainline/model/policy.py, local g3_rollout slice and both
  static consumers.
- caa7e33:clearvla/mainline/model/transition.py,
  ControlledTransitionCompiler.build_source.

Schema25 defect:

- transition.py rebuilds four interval groups from
  DenseFactChart.public_scene_base and a learned interval_identity;
- the transition therefore receives a new learned temporal identity instead
  of the exact completed G3 rows already read by P1.

Closed boundary:

    completed grounding canvas
      -> exact rollout slice [B, 4*C*8*8, H]
      -> static P1 and ControlledTransitionSource.selector
      -> dynamic transition coefficients at each Euler node
      -> bottom/event/action losses

The static tensor is built once per observation and reused at the five dynamic
nodes. Action gradients can accumulate through every use back to the same G3
producer. No mean, expand or learned interval reconstruction is legal at this
handoff.

Replay decision:

- preserve the Schema26 tensor identity and shape check;
- remove only transition-owned interval_identity;
- do not import unrelated Schema26 S operators with this unit.

Static acceptance:

- P1 and transition receive the same source tensor object or an exact view;
- a sentinel perturbation of one G3 row reaches the matching transition row;
- no expand recreates the interval axis;
- forward and reverse VJP traces reach the G3 producer.

Unresolved assumption: none at the semantic boundary. Parameter migration is
intentionally rejected for the replay candidate.

### G-02: conditional-K G3 and exported-content reconstruction

Source anchors:

- 2a0d3d1:grounding.py, dense_chart_from_local_facts and typed prebinding.
- 127fee8:grounding.py, _conditional_k_reconstruction_assignment and the
  null-independent reconstruction path.
- 03235d3:grounding.py, physical K/null ownership and typed refinement inside
  physical support.

Schema25 defects that exist:

- DenseFactChart.dino_content is rebuilt from online content_slots, so a
  self-consistent collapsed online chart can also become its own target.
- G3 corrects K and null together, so a learned residual can change absolute
  object-versus-null mass when only conditional K refinement is intended.
- reconstruction contains a private object-specific content residual and can
  reduce loss without placing all useful content in ObjectFactSet.content.

The least restrictive R1 boundary is narrower than the complete Schema35
rewrite:

    detached current DINO chart
      -> observable target only

    existing Schema25 physical K+null binder
      -> preserve real/null mass
      -> G3 changes only P(K | real)
      -> conditional-K * local prior * observable validity
      -> reconstruction assignment independent of learned null
      -> exported ObjectFactSet.content is the only K-specific value

R1 deliberately retains the Schema25 base binder inputs. It does not yet adopt
Schema27 equal typed-logit consensus or Schema35 content-only identity. Which
observable views should establish K identity remains a separate unit; fixing
the real/null and reconstruction ownership does not require deciding it.

Axes and numerics:

- conditional owner [B,N,K] remains FP32;
- local-hypothesis prior and observable validity stay outside its softmax;
- all-invalid candidates produce zero assignment;
- the target is detached;
- no separate learned-null value or gate appears; reconstruction assignment
  and its Jacobian are null-free. The one exported online content remains a
  legal product of the retained physical K+null binder.

Gradients:

- reconstruction reaches the physical assignment and the same exported K
  content consumed by S, W and Teacher;
- no gradient reaches the detached DINO target;
- action/future losses continue through ObjectFactSet consumers independently
  of the reconstruction objective.

Rejected mechanics:

- equal averaging of semantic, appearance and geometry logits as a new K
  identity rule;
- a private K-specific decoder that is visible only to the loss;
- a learned null that can switch off reconstruction;
- a new fixed reconstruction gain.

Static acceptance:

- intervening on the retained slot residual changes exported K content and
  reconstruction through that same value;
- no loss-private second K-specific value remains;
- changing null logits while holding conditional K fixed leaves the
  reconstruction assignment unchanged;
- conditional K sums to one on valid real support and stays zero for invalid
  rows.

### S-01: one typed owner and no CoarseAction typed re-entry

Source anchors:

- caa7e33:intent.py, StatelessObjectIntentOrganizer._typed_relevance and
  CoarseActionIntent.forward.
- caa7e33:types.py, ActionIntentDock and ObjectIntentState.
- e7d7f42 is an anti-donor because it recreates a typed CoarseAction route;
  c159651 removes that route again.

Schema25 defect:

- S-owned typed evidence reaches W directly and also reaches W indirectly
  through CoarseActionIntentState;
- the same semantic owner therefore appears under two names before W.

Replay boundary:

    language/history/current ObjectFactSet
      -> S-owned typed relevance [B,I,K,type,R]
      -> named WorldIntentDock
      -> W once

    public S + public object/history context
      -> CoarseAction
      -> W action condition

CoarseAction may consume public observable intent but cannot consume the typed
relevance value or recreate a typed selector. P2 may later consume S metadata
only after a W-owned spatial selection; that is a named conditioning consumer,
not a second value owner.

R1 preserves the Schema25 relevance scoring operator. Schema26's separate
common/differential floor and cap are not required for this ownership repair.

Static acceptance:

- no typed field exists in ActionIntentDock or CoarseActionIntentState;
- exactly one typed S value reaches the W producer;
- zero typed S leaves CoarseAction unchanged;
- W action and S gradients remain ordinary and distinct.

### S-02: lossless common and interval-residual views

Source anchors:

- c159651:intent.py and types.py, separate typed_common and typed_interval
  residual values.
- 1b11bf5:intent.py, conditional-K reductions and preserved factual intent
  identities.

Purpose:

- W needs a protected shared typed value and a genuine interval innovation;
- recomputing common by averaging a value after W would let downstream
  processing rewrite its owner;
- storing only a residual and later expanding an inferred common loses the
  original identity.

R1 uses an algebraically lossless decomposition of the existing Schema25 S
value:

    common = mean over the four source interval rows
    residual_i = source_i - common
    source_i = common + residual_i

Both tensors are retained. The mean is a decomposition coordinate, not a
fusion gain: it is never used to attenuate a lone active type and the original
source can be reconstructed exactly.

R1 does not import later fixed variance floors, fixed type divisors, labelled
stage identities or signed tanh scoring. Those are independent mechanism
choices and remain open.

Static acceptance:

- reconstruction error of source from common plus residual is exact zero
  within dtype tolerance;
- residual sums to zero over the source interval axis;
- K and type axes remain present;
- no expand manufactures a missing K or type axis.

Implementation closure: R1c is recorded in
`R1C_S01_S02_TYPED_INGRESS_DECOMPOSITION_WORKSHEET.md`. The existing
Schema25 score is decomposed only after it is computed; typed common/residual
fields enter W through `WorldIntentDock`, ActionIntentDock is typed-free, and
the R1c W boundary reconstructed its former source once. R1d later supersedes
that temporary consumer mechanic while preserving the same sole ingress. The
S-03 no-future-owner guard also passes. The retained suite is 134/134 with no
parameter, optimizer or state-key addition.

### S-03: future-owner supervision fence

Source anchors:

- 2a0d3d1:intent.py, DirectIntentFutureSupervisor.
- 4ac7e54:intent.py, ObservableIntentStateSupervisor.

Schema25 has no direct pre-W semantic/transport/status prediction head.
DirectIntentFutureSupervisor is therefore not a repair required by the replay
base. It gives S/W-boundary projections the same targets later owned by W and
creates an identity-W shortcut.

R1 keeps the Schema25 training boundary and rejects the direct future-field
addition. If the existing Schema25 recognizer is changed later, the maximum S
auxiliary scope is an observable adjacent-state target; it cannot decode W
semantic, geometry, transport, status or physical-support targets.

This is a guard rather than a new implementation slice.

### W-01: one-way common-to-interval ownership

Source anchors:

- 4ac7e54:dynamics.py, _run_owned_typed_block.
- 4363108:dynamics.py, _run_separated_owned_typed_block, used as an anti-donor.
- current dynamics.py, _condition_interval_on_common, used only to inspect the
  surviving producer and consumer boundary.

Required causality:

    protected common
      -> processed once
      -> may condition each present interval innovation
      -> cannot be rewritten by an interval residual

    interval residual_i
      -> reads protected common
      -> remains owned by interval i
      -> W2 may read near residuals to write far residuals

Schema34's bilateral separation is rejected because it also forbids the legal
common-to-residual relation. Concatenating common as the first causal token is
a donor mechanism, not a mandatory literal implementation. R1 may use an
explicit zero-preserving conditioner if that makes call ownership and W2
repetition easier to prove.

Call contract:

- W1 owns common and near intervals;
- W2 cannot rewrite common or near and writes only far intervals;
- common is not reprocessed at W2;
- the public field is formed once from common plus each residual.

Static acceptance:

- residual zero returns the processed common field;
- common zero cannot create a hidden common value through bias;
- perturbing a far residual cannot change common or near outputs;
- reverse tracing from every final interval reaches the one common producer
  and its matching residual producer exactly once.

### W-02: typed W field and online ABI cleanup

Source anchors:

- 2a0d3d1:dynamics.py, typed semantic/appearance/geometry sidecars.
- 03235d3:dynamics.py and types.py, observable chart availability and
  camera-preserving transport/covariance.
- 1b11bf5:dynamics.py and types.py, removal of visibility and persistence
  action values and the final supervised FutureObjectDynamics ABI.

R1 output ownership:

- semantic common/residual retains [B,I,K,H];
- appearance is a zero-preserving conditioner of semantic state, not an
  independent online status value;
- transport and covariance retain [B,I,K,C,*] until the geometry consumer;
- covariance is FP32 PSD and may approach zero;
- only the final supervised FutureObjectDynamics crosses W into P2.

Schema25 predicted visibility, persistence, uncertainty, reliability and
future_selector_validity are removed from the online action ABI. They cannot
scale P2 support or become an action value. Teacher association null remains a
training-plane identity fallback and audit, not physical disappearance.

R1 retains the Schema25 Teacher row-softmax backend. Removing status authority
does not require adopting Schema31 partial OT.

Gradients:

- W future objectives supervise semantic, transport and covariance owners;
- action/event/motion gradients reach W only through P2, consequence and
  bottom;
- Teacher targets remain detached and deployment still runs Teacher zero
  times.

Static acceptance:

- setting observable chart support to zero makes the matching W/P2 value zero;
- changing a predicted status tensor is impossible because it is absent from
  the online ABI;
- camera perturbations survive to the geometry consumer without pre-averaging;
- neutral W heads make the downstream effect exact zero.

Implementation closure: R1d is recorded in
`R1D_W01_W02_CAUSAL_FIELD_ABI_WORKSHEET.md`. W1 processes the protected common
once and writes only near innovations; W2 reads completed W1 and writes only
far innovations. Appearance and generic context are zero-preserving
conditions, not value owners. The existing camera transport-prior projection
creates distinct real-camera carriers without a new parameter. Teacher keeps
the Schema25 row softmax while exporting per-camera null-identity moments;
future losses and the minimum P2 adapter consume C before reduction. Predicted
status/support fields, heads, losses and P2 values are absent. The retained
suite is 140/140. The net inventory delta is -3,075 parameters and -7
parameter/state keys with all 23 optimizer groups preserved. The inherited
flattened `[I,K]+null` and semantic/geometry type terminal remain explicit
P2-01 debt.

### P1-01: static fact versus dynamic policy residual

Source anchors:

- 03235d3:types.py and restored_bottom.py, CompletedP1PolicyState.
- 6bc6218 and eac4916:compiler.py, transition.py and
  time_domain_mmdit.py, dynamic precision consumers.

Schema25 defect:

- static high-resolution factual detail and a much larger action/time
  dependent write are added and then passed under one factual identity;
- P2, P3, transition and bottom cannot tell observation evidence from live
  policy refinement.

R1 boundary:

    static P1 once
      -> factual_base [B,T,Q,H]

    noisy action + time + factual_base at each dynamic call
      -> policy_query_residual [B,T,Q,H]

    P2 query = action_query + factual_base + policy_query_residual
    protected_policy_precision = policy_query_residual

R1 does not require Schema39's additional fact/action multiplicative
interaction around the dynamic residual. The residual is already policy
conditioned. A later zero-preserving interaction can be a separate capacity
unit if a source-level consumer need is demonstrated.

The protected dynamic carrier has no learned null. It is consumed once by the
transition action operand and once inside the bottom's single optional ingress
site. It never enters factual memory or protected consequence.

Static acceptance:

- zero dynamic residual leaves factual_base and consequence unchanged;
- factual_base is invariant to noisy action and Euler time;
- both legal dynamic consumers receive the same carrier;
- no optional P3 lane receives `policy_query_residual`; removal of inherited
  optional factual/static-precision aliases is owned by P3-01.

Implementation closure: R1e is recorded in
`R1E_P101_STATIC_DYNAMIC_P1_WORKSHEET.md`. The exact static reader is unchanged
and still runs once. `CompletedP1PolicyState` keeps its factual base and live
policy residual disjoint; only `P2QueryDock` forms the exact three-term P2
query. Consequence starts from the static fact. The same raw residual reaches
the controlled-transition action operand and the retained bottom optional
ingress through the existing no-null basis reader, without a new parameter,
contract, null or scale. Forward/reverse boundary tests and the retained suite
pass 140/140. R1e has zero parameter, optimizer and state-key delta from R1d;
P2-01 is the next source unit.

### P2-01: spatial selection then physical interval terminal

Source anchors:

- a2b5705:compiler.py, factorized interval/object structure; exact shared
  posterior is an anti-donor.
- 6bc6218:compiler.py, complete W field consumption and S-conditioned W key.
- eac4916:effect_terminal.py, SelectedIntervalEvidence, spatial_select and
  temporal_terminal.

Schema25 defect:

- interval and object are flattened into one competition with a learned null;
- predicted W visibility controls support;
- semantic, transport and status then compete through a type softmax;
- the physical interval axis is destroyed at P2.

R1 boundary:

    action query + supervised W field + observable G support
      -> semantic K selection for every interval
      -> geometry K*C selection for every interval
      -> SelectedIntervalEvidence retaining [B,T,Q,I,type,H]

    selected S metadata from the W-owned spatial posterior
      -> zero-preserving condition of the selected W key
      -> one four-physical-interval terminal per type
      -> common once + posterior-weighted interval residual
      -> semantic plus geometry complementary sum
      -> one inherited P2 boundary contract

There is no learned null at the physical interval terminal. Empty observable
support returns zero. S cannot create K/camera support, a value or an
independent interval logit.

R1 uses the Schema38 semantic relation, not the Schema39 score-level triple
product:

    conditioned_key = W_key + W_key * bounded_function(S_context)
    interval_score = relation(action_query, conditioned_key)

The exact bounded function and normalization floor are implementation details
to be selected without a new fixed gain. The mandatory property is
zero-preserving key conditioning: neutral W stays neutral, while S can change
the W-owned relation before the terminal.

Rejected mechanics:

- Schema27 subtraction of log(I*K);
- Schema33 subtraction of log(K) and one shared type-time posterior;
- Schema29/30 division by 3 or square-root 3;
- Schema38 fused object/time/null terminal;
- Schema39's exact attenuating score formula;
- a type softmax over complementary semantic and geometry values.

Static acceptance:

- K, camera, interval and type disappear only at their named consumers;
- a one-interval sentinel cannot move to another interval before the terminal;
- no learned or predicted validity controls physical amplitude;
- zero W gives zero effect for all S/action inputs;
- semantic-only and geometry-only legal inputs survive independently;
- complementary fusion uses the existing single outer contract and adds no
  per-type gain.

Implementation closure: R1f is recorded in
`R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md`. Semantic K and geometry K*C
posteriors are normalized independently for every physical interval and use
only current observable support. `SelectedIntervalEvidence` preserves I/type
and the exact common-plus-innovation identity. The same W posterior selects S
metadata, which conditions only the selected W key through the adopted
zero-preserving relation. Each type then owns one no-null four-interval
terminal; their raw latent contributions add before the one inherited caller
contract. All-invalid rows are finite exact zero. The R1f retained suite passed
144/144; R1g/P3-01,B-01 closure is recorded below.

### P3-01: unique P3 values around protected carriers

Source anchors:

- 2a0d3d1 and e7d7f42:compiler.py, factual/effect deduplication history and
  consequence-conditioned temporal recovery.
- 1b11bf5:compiler.py, later six-lane expansion used as an alias audit.
- eac4916:compiler.py, protected_policy_precision.

R1 value inventory:

1. protected_consequence: factual_base plus typed effect plus zero-preserving
   factual/effect interaction;
2. protected_policy_precision: dynamic P1 residual;
3. optional temporal innovation;
4. optional observable state-change innovation.

There is no optional factual lane, optional static precision lane, or optional
semantic/geometry effect lane. Those values already have protected owners.

The temporal lane may read S temporal context and the consequence innovation
effect plus interaction, but it cannot reproject the complete protected fact.
The state-change lane reads its observable S-private operand. Both remain
bias-free and exact zero for zero private input. R1 removes Schema25's fixed
state-change multiplier and does not introduce a replacement per-lane gain;
the existing single bottom ingress contract owns the outer scale.

Static acceptance:

- each lane has a named private operand;
- zero W returns protected consequence exactly to factual_base;
- zero dynamic P1 returns protected_policy_precision exactly zero;
- optional lane zero remains zero through routing;
- no tensor reaches bottom twice under different semantic names.

### B-01: lane-local bottom reads and protected no-null reads

Source anchors:

- 1b11bf5:v120_core/time_domain_mmdit.py,
  _read_policy_delta_bank with per-lane four-basis-plus-null reads.
- eac4916:v120_core/time_domain_mmdit.py, no-null protected dynamic precision.

Schema25 defect:

- five semantic lanes times four action bases enter one shared competition
  with one null;
- unrelated lanes suppress one another before their meanings are consumed.

R1 boundary:

- temporal and state-change each receive an independent shared-parameter
  four-basis-plus-zero-null read;
- protected consequence receives its existing no-null basis read;
- protected dynamic precision receives one no-null basis read;
- all optional and dynamic-precision values meet at the one existing optional
  ingress site before the retained bottom.

R1 adds no second amplitude budget and no lane quota. Existing outer bottom
scales are inherited as base mechanics, not endorsed as final truth; changing
them is a separate unit.

Static acceptance:

- increasing one lane's null logit cannot change another lane's posterior;
- protected carriers have no null parameter;
- the bottom write call count remains one per dynamic node;
- all retained V120 CVAE/workspace/controller consumers remain reachable.

Implementation closure: R1g is recorded in
`R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md`.
`ObjectPolicyPlanDeltaBank` now carries only protected consequence, raw dynamic
precision, temporal and state-change. Temporal uses only S temporal context and
the consequence innovation under action conditioning; state-change preserves
its S-private multiplicative zero. The three optional protected-value aliases,
fixed `0.05` multiplier and `sqrt(2)` attenuation are absent. Bottom invokes
one shared Q+zero-null reader separately for the two optional lanes, adds their
raw reads, then joins a separate no-null precision read before the existing
fixed optional scale. Consequence remains a separate no-null write outside the
scale. The manifest names both changed ABIs, retained/downstream seed-0
initialization remains R1f-identical, and the retained suite passes 145/145.

LC-01 is the closed deletion-only cleanup recorded in
`LC01_EXACT_ZERO_LAYER_TRAJECTORY_CLEANUP_WORKSHEET.md`. A complete
producer-to-consumer and reverse-VJP audit proved that the two inherited
layer-contract trajectory formulas and their frozen action/motion probes have
exact-zero effect on every live contract output and active loss. They are now
absent; both independent terminal rollout/state/event adapters remain. The
cleanup deletes 23,590 frozen parameters and 16 state keys while preserving
all retained contract/decoder tensors, the optimizer partition and the seed-0
construction RNG exactly.

### N-01 and D-01: numerical support and matching diagnostics

Source anchors:

- 6bc6218 grounding/compiler and gradient_audit.py.
- eac4916 grounding.py, effect_terminal.py and gradient_audit.py.

These are support units, not representation capacity:

- observable probability measures and log measures remain FP32;
- all-invalid masked softmax returns finite exact zero;
- variance-to-standard-deviation conversion is zero preserving;
- diagnostics read detached tensors and cannot alter loss, support or routing;
- spike attribution names the actual owning parameter and separates channels
  only when that exact producer has the matching channel ABI.

Only diagnostics whose producer still exists in R1 are copied. Historical
metrics are not recreated merely to preserve a log vocabulary.

Implementation closure: R1h is recorded in
`R1H_N01_D01_FINITE_NUMERICS_DIAGNOSTICS_WORKSHEET.md`. The four active address
variance consumers now use the finite-slope exact-zero transform. G2/G3 retain
one FP32 probability/log view of their existing owner measure through the
grounder, W/Teacher boundary and P2; no support floor or new owner is present.
Source-gradient hooks are identity VJPs, finite-spike scans are rare-path and
pre-clip only, and the runtime vocabulary names live R1 producers. The LC-01
parameter/state/optimizer/tensor/RNG fingerprints remain exact and the retained
suite passes 155/155.

## 4. Deferred and rejected donor mechanics

| Donor mechanic | Decision | Source reason |
|---|---|---|
| Schema27 fixed set-versus-null correction by subtracting log(I*K) | REJECT | Candidate count becomes an architectural prior over absence |
| Schema27 direct pre-W semantic/status/transport supervision | REJECT | Duplicates W's future owner and creates an identity-W shortcut |
| Schema27 predicted visibility as selector validity | REJECT | Gives a prediction authority over physical support and value amplitude |
| Schema28 typed CoarseAction route | REJECT | Reintroduces the second S-to-W typed path removed in Schema26 |
| Schema29 protected mean plus tiny contrast residual | REJECT exact form | Complementary owners are attenuated by an assumed active-owner count |
| Schema30 division by square-root 3 | REJECT | Fixed active-owner assumption remains |
| Schema31 partial OT/dustbin backend | DEFER | Changes the Teacher target distribution; not required for R1 ownership closure |
| Schema32 zero-initialized typed-base gain | REJECT as invariant | Zero preservation is required; a fixed opening mechanic is not |
| Schema33 shared temporal posterior and subtraction of log(K) | REJECT | Types are forced to share time and candidate count calibrates null |
| Schema34 bilateral common/residual isolation | REJECT | Blocks legitimate common-to-residual conditioning |
| Schema35 content-only K identity | DEFER | Physical owner is plausible, but selecting its observable evidence is independent of G3/null repair |
| Schema37 six optional P3 lanes | REJECT exact form | Static precision and typed effect aliases repeat protected owners |
| Schema38 fused object/time/null effect reader | REJECT terminal form | Destroys the physical interval axis at the spatial consumer |
| Schema39 exact action-times-S-times-W score | REJECT exact form | Ownership is legal but the relation contains avoidable hidden-width attenuation |
| Any quota, entropy target, forced non-null reward or synthetic gradient | REJECT | Changes optimization pressure instead of repairing ownership |

## 5. Dependency and conflict graph

    G-01 -----------------------------------------------> retained transition

    G-02 -> S-01 -> S-02 -> W-01 -> W-02 -> P2-01
                         \                         /
                          \-> CoarseAction public /

    static P1 -----------------------> P1-01 ----/

    P2-01 -> protected consequence -> P3-01 -> B-01 -> retained bottom
                    P1-01 precision -----^        |
                                                  -> action/event/motion loss

    Schema25 Teacher row softmax -> W-02 objectives
    N-01 supports G-02, W-02 and P2-01
    D-01 observes the closed graph only

Hard conflicts:

- S-01 conflicts with the Schema28 typed CoarseAction bypass.
- W-01 conflicts with Schema34 bilateral isolation.
- W-02 conflicts with online status/visibility selector authority.
- P2-01 conflicts with Schema27/33 fixed null calibration and the Schema38
  fused terminal.
- P3-01 conflicts with Schema37's six-lane alias inventory.
- B-01 conflicts with Schema25's joint lane-by-basis simplex.

## 6. R1 candidate boundary

Working name:

    Schema25-R1 ownership and terminal closure

R1 is one training candidate assembled as reversible source commits. It is not
one experiment per unit and is not a recreation of Schema26, Schema32,
Schema37 or Schema39.

Implementation slices:

1. G exact handoff: G-01.
2. G null/reconstruction closure: G-02.
3. S docks and decomposition: S-01, S-02 and S-03 guard.
4. W causal field and ABI: W-01 and W-02.
5. static/dynamic P1: P1-01.
6. spatial and physical terminal: P2-01.
7. unique P3 and bottom ingress: P3-01 and B-01.
8. matching numerics and diagnostics: N-01 and D-01.

Slices 1-8 are complete as separate reversible source units. P2-01 closure is
recorded in `R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md`: K/C disappear
  only at their type-local spatial consumers, I disappears only at independent
  no-null physical terminals, and semantic/geometry add before the single outer
  contract. P3-01/B-01 closure is recorded in
  `R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md`: protected values have
  no optional aliases, and the two real optional innovations receive disjoint
  normalization calls. N-01/D-01 closure is recorded in
  `R1H_N01_D01_FINITE_NUMERICS_DIAGNOSTICS_WORKSHEET.md`: numerical support is
  finite without creating mass, and diagnostics observe the closed graph only.

There is no training between these slices. Each slice receives only algebraic,
shape, sentinel, call-count, forward-trace and reverse-VJP verification. The
limited training budget is considered only after the entire R1 source graph
closes.

R1 deliberately freezes:

- Schema25 Teacher row-softmax association backend;
- Schema25 S relevance scoring before the lossless common/residual view;
- Schema25 physical K binder inputs;
- retained V120 bottom, CVAE/workspace, controller and execution lifecycle;
- one existing outer P2/bottom amplitude boundary, with no new per-owner gain.

R1 deliberately does not freeze:

- a fixed relevance floor, temperature, cap or entropy;
- a fixed complementary-owner divisor;
- a null prior derived from candidate count;
- bilateral common/residual isolation;
- a learned visibility/existence amplitude gate;
- Schema39's exact S-W scoring relation;
- Schema31 partial OT.

## 7. Historical preconditions used before R1 source implementation

The semantic units were source-acquired under an implementation-boundary
worksheet for the exact replay checkout. Before each R1 slice began, the
implementation task had to:

1. check out or create the Schema25 replay branch without mutating the
   then-current Schema39 worktree;
2. map the exact touched producer and all consumers in that checkout;
3. record optimizer ownership, state-dict keys, config/manifest fields,
   deployment call sites and diagnostics;
4. record unresolved mechanism choices in the slice commit message;
5. reject exact checkpoint migration and plan a fresh formal checkpoint.

No source unit was considered closed merely because its historical donor tests
passed on Schema39. This section records the completed R1 procedure; it is not
an instruction to recreate or modify a current checkout.
