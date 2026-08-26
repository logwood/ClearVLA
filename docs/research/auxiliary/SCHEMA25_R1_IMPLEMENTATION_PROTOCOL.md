# Schema25-R1 implementation protocol

Status: adopted implementation discipline; R0 fingerprint complete; R1a/G-01
through R1g/P3-01,B-01 plus LC-01 exact-zero cleanup implemented and
statically closed; R1h/N-01,D-01 next; no training run.

This file reconciles the user-supplied
C:/Users/ASUS/Desktop/ClearVLA_Schema25_Replay_Implementation_Protocol.md
with the adopted cross-version replay plan. The supplied document is reference
material, not an executable command list.

Supplied-document SHA-256:
1FA06243FAC8EA45704FDCE5BD1A57B17A45700C6C08E69D4292F1B4C10B4B20.

## 1. Authority and scope

Authority order inside this replay worktree:

1. ARCHITECTURE_REPLAY_PLAN.md selects the base, dispositions and R1 bundle.
2. ARCHITECTURE_REPLAY_SOURCE_UNITS.md defines source boundaries,
   dependencies, conflicts and rejected mechanics.
3. This file defines implementation sequencing and release-blocking checks.
4. The supplied external protocol remains provenance for the implementation
   discipline but cannot independently add mechanisms, run experiments, load
   checkpoints or change the bundle.

R1 is not a recreation of Schema37, Schema38 or Schema39. It reconstructs the
selected cross-version semantic units on the exact Schema25 source:

    base:       6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6
    branch:     codex/schema25-r1-replay
    worktree:   .worktrees/schema25-r1-replay
    candidate:  Schema25-R1 ownership and terminal closure

No architecture mechanism may be invented during implementation. A required
mechanism absent from the source-unit contract stops the slice and returns it
to specification review.

## 2. Corrections to the supplied protocol

- Historical schema numbers are donor coordinates, not implementation phases.
- R0 fingerprinting is static and local by default. No CUDA smoke, training
  step or dataset access is implied by worktree creation.
- Schema25 checkpoints are not migrated into R1. Structural ABI changes require
  a fresh formal checkpoint.
- A micro-counterfactual uses one fixed R1 initialization and one synthetic or
  otherwise explicitly authorized batch. It does not coerce an incompatible R0
  checkpoint into the R1 graph.
- Intervention tests prove reachability, locality and forbidden-path absence.
  They do not impose a minimum effect magnitude or usage quota.
- Existing outer contracts may be retained as base mechanics. A replay slice
  cannot add a second normalization, gain, floor or amplitude budget.

## 3. Untouched R0 baseline

Before replay source changes, record:

- commit and tree hashes;
- clean source status before replay documents are imported;
- manifest payload and digest;
- tracked source/config/lock fingerprints;
- total, trainable and top-level module parameter counts;
- optimizer role and group counts;
- static and lightweight checks;
- selected Schema25 structural tests;
- Teacher deployment call contract from source and tests.

R0 does not require a log reread, checkpoint, dataset or training run. Replay
documents are documentation-only additions and are separated from the R0
model-source tree hash.

## 4. Locked mechanism contracts

These contracts constrain implementation. Exact symbol placement is resolved
by the mandatory per-slice producer/consumer worksheet before editing.

### 4.1 G handoff and reconstruction

The completed G3 rollout is an exact view shared by static P1 and transition.
No mean followed by expand and no transition-owned interval identity are
allowed.

G3 may change only conditional K identity:

    P(real)                     preserved
    P(K | real)                 corrected by G3
    P(null)                     preserved
    reconstruction assignment  P(K | real) * local prior * observable validity

The reconstruction target is detached current DINO. The only K-specific
reconstruction value is the same exported object content consumed by S, W and
Teacher. The existing slot residual may remain only by becoming part of that
exported value; no loss-private K decoder and no learned-null reconstruction
gate are allowed.

### 4.2 S single ingress and decomposition

R1 retains the Schema25 typed relevance scoring mechanism:

    source_i   = Schema25 S typed relevance value
    common     = mean_i(source_i)
    residual_i = source_i - common
    source_i   = common + residual_i

The mean defines a lossless coordinate decomposition; it is not a
complementary-owner fusion or magnitude divisor.

Typed S reaches W through one named dock. CoarseAction receives only public
observable S/object/history context and cannot carry typed relevance under a
new name.

### 4.3 W one-way ownership

The donor token sequence was reviewed but is not copied literally:

    [common, residual_0, ..., residual_n]
      -> existing causal W block
      -> processed_common, processed_residual_0, ..., processed_residual_n

Although its mask protects common from later residuals, attention also turns an
exact-zero residual into a nonzero value by reading common. That violates the
adopted rule that conditioning cannot manufacture the interval owner. R1d
therefore resolves the same one-way causality with the parameter-free relation
`x + x * tanh(c)`: W1 processes common as one singleton interval exactly once;
near innovations may be conditioned by completed common/generic context, and
W2 uses completed common/near only as a conditioner of present far
innovations. A zero innovation remains exact zero. No projection, learned
gain, floor or bilateral isolation is added.

W1 processes common once and owns near intervals. W2 reads the completed near
state and writes only far intervals; it cannot process common a second time or
rewrite near.

The closed W field contains semantic delta/successor plus camera-specific
transport and FP32 PSD covariance. Appearance conditions semantic state and
has no independent status value. C comes only from the existing current-camera
transport prior and remains real through Teacher targets, future losses and
the P2 geometry consumer. Observable current chart/camera availability is the
only online support; predicted visibility, persistence, uncertainty,
reliability and selector validity are absent. Teacher retains the Schema25
candidate-plus-null row softmax and allocates null identity per camera after
association. R1f/P2-01 now keeps I through semantic K and geometry K*C
selection, then uses one no-null physical-I terminal per type. It cannot retain
status or reduce C before covariance/transport consumption.

### 4.4 S conditioning of the W-owned terminal relation

The allowed relation is a zero-preserving key conditioner:

    s = tanh(existing_bounded_view(selected_S_context))
    conditioned_W_key = selected_W_key + selected_W_key * s
    interval_score = existing_action_key_relation(action_query, conditioned_W_key)

Zero S leaves the W key unchanged. Zero W remains zero for every S/action
input. S cannot create support, value, a spatial posterior or an independent
time vote.

The implementation may reuse a source-existing projection only when its
producer and consumer meanings match. It may not add a new gain, minimum gain,
floor, quota or the Schema39 score-level triple product.

### 4.5 P1 and P3

    static factual_base         built once, observation-owned
    dynamic policy residual    built at each dynamic call

    P2 query = action_query + factual_base + dynamic_policy_residual
    protected_policy_precision = dynamic_policy_residual

The dynamic residual has protected accessibility, not protected magnitude. It
may be exact zero and receives no learned-null competition.

P3 retains protected consequence, protected dynamic policy precision, optional
temporal innovation and optional observable state-change innovation. It removes
optional factual, static-precision and semantic/geometry-effect aliases.

The temporal private source is S temporal context plus the projected W effect
and factual/effect interaction. It does not reproject the complete factual
consequence and has no fixed two-source divisor. Action conditioning and the
one inherited outer lane contract remain.

### 4.6 Complementary fusion and bottom

Semantic and geometry are complementary:

    selected semantic + selected geometry
      -> one inherited P2 outer contract

There is no division by 3, division by square-root 3, type softmax or second
amplitude budget.

Temporal and state-change each receive an independent four-basis-plus-zero-null
read. Protected consequence and protected dynamic precision use no-null basis
reads. All optional values enter the retained bottom at one existing aggregate
ingress.

## 5. Test-first slice order

No training occurs between slices.

1. R1a / G-01: exact G3 source identity.
2. R1b / G-02: conditional-K and exported-content reconstruction.
3. R1c / S-01,S-02: typed dock cleanup and lossless decomposition.
4. R1d / W-01,W-02: causal common/near/far field and online ABI.
5. R1e / P1-01: static/dynamic P1 separation.
6. R1f / P2-01: spatial selection and physical interval terminal.
7. R1g / P3-01,B-01: unique lanes and lane-local bottom ingress.
8. R1h / N-01,D-01: matching finite numerics and diagnostics.

Execution ledger:

| Slice | State | Evidence |
|---|---|---|
| R1a / G-01 | `COMPLETE` | `R1A_G01_G3_HANDOFF_WORKSHEET.md`; 123/123 retained tests |
| R1b / G-02 | `COMPLETE` | `R1B_G02_CONDITIONAL_K_RECONSTRUCTION_WORKSHEET.md`; 129/129 retained tests |
| R1c / S-01,S-02 | `COMPLETE` | `R1C_S01_S02_TYPED_INGRESS_DECOMPOSITION_WORKSHEET.md`; 134/134 retained tests |
| R1d / W-01,W-02 | `COMPLETE` | `R1D_W01_W02_CAUSAL_FIELD_ABI_WORKSHEET.md`; 140/140 retained tests |
| R1e / P1-01 | `COMPLETE` | `R1E_P101_STATIC_DYNAMIC_P1_WORKSHEET.md`; 140/140 retained tests; zero parameter/state delta |
| R1f / P2-01 | `COMPLETE` | `R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md`; 144/144 retained tests |
| R1g / P3-01,B-01 | `COMPLETE` | `R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md`; 145/145 retained tests |
| LC-01 exact-zero cleanup | `COMPLETE` | `LC01_EXACT_ZERO_LAYER_TRAJECTORY_CLEANUP_WORKSHEET.md`; retained contract/decoder/RNG exact; 145/145 retained tests |
| R1h / N-01,D-01 | `NEXT` | close the numerical-support/diagnostic worksheet before source edits |

For every slice:

1. finish its exact producer/transformation/consumer/loss/runtime/checkpoint
   worksheet;
2. add failing structural/intervention tests;
3. implement only the named unit;
4. run forward and reverse boundary review;
5. run relevant tests and then the retained structural suite;
6. record source fingerprints and unresolved assumptions;
7. commit one reversible semantic unit.

## 6. Release-blocking static matrix

R1 cannot proceed to a formal run unless all applicable rows pass.

| Boundary | Required evidence |
|---|---|
| G3 handoff | P1 and transition consume the exact completed tensor/view; no reconstructed interval axis |
| Reconstruction | conditional K is normalized; invalid rows are zero; learned null cannot alter its assignment/Jacobian |
| S ingress | no typed field in ActionIntentDock or CoarseAction output; one named typed S-to-W path |
| S decomposition | common plus residual equals source; interval residual sums to zero; K/type retained |
| W ownership | residual cannot change common; far cannot change near/common; no minimum conditioning magnitude |
| W ABI | status/predicted-validity cannot control P2 support; camera survives to geometry consumer |
| P1 | factual base invariant to noisy action/time; dynamic residual remains separately observable |
| P2 | K/C/I/type persist until named consumers; zero W produces exact-zero effect |
| P3 | no factual/effect/static-precision duplicate owner; each optional lane has one private operand |
| Bottom | one lane null cannot change another lane; protected carriers have no null |
| Runtime | static/dynamic call counts match lifecycle; deployment Teacher calls and tensor reads are zero |
| Gradients | legal downstream VJPs reach the named owner; forbidden and duplicate bypasses are absent |

Tests verify algebra and graph closure, not task quality. No assertion requires
a learned route to exceed a nonzero magnitude threshold.

## 7. Micro-counterfactual boundary

After structural closure and before any training decision, use one fixed R1
initialization and the same authorized input:

    baseline
    zero/shuffle S-private terminal
    zero/shuffle W consequence
    zero/shuffle dynamic P1
    zero each optional P3 lane

Record downstream tensor locality and action reachability. Exact zero at a
supposed unique consumer reveals a missing path; a minimum delta is not
required.

This stage is skipped if constructing the input would consume dataset,
checkpoint or experiment budget not separately authorized.

## 8. Formal-run boundary

Before a formal run, generate an immutable run context containing:

- commit, tree and source fingerprints with a clean worktree;
- replay parent and ordered semantic commits;
- manifest and structural-test fingerprints;
- dataset, split and normalizer identities;
- Python, NumPy, Torch, CUDA, sampler and DataLoader seeds;
- optimizer, schedule, precision and clipping configuration;
- total/trainable/frozen/module/optimizer-role parameter counts;
- fresh checkpoint with no resume state;
- Teacher implementation identity and deployment-disabled assertion;
- hardware and runtime versions;
- structural and numerical stop rules fixed before launch.

Performance is not a substitute for structural correctness, and no
performance-driven architecture stop rule is invented during a run.
