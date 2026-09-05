# Current ClearVLA architecture contract

Updated: 2026-09-05

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
optional hybrid:        32; explicit hybrid-v1 Pen training/solver contract below
layout schema:          2 (atomic modular owner layout)
manifest digest:        c6742c2c5a8a381193a56ed2cf24ba632b472cfde168c021537f2d30b8d69863
active source identity: Schema28-core recovery plus profile-owned gripper boundary, CALVIN direct relative-command arm chart and binary-command isolation
historical Schema30 source: 3fef2fc0dce297f600c813307c998f587cca1ca3
formal CALVIN checkout:  f9cee96a14dfb328830aa5c1fd4fd54cb33d4181
branch:                 codex/schema29-mainline (historical branch name only)
behavior reference:     Schema28, commit 097330a894d948d66c419f8af07325a5b0ff712e
recovery reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global objects:         K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0; two adjacent learned flows
training:                fresh, single-stage end-to-end; recovery uses one formal forward/loss pass
bottom:                 V120 seed/transition/CVAE/workspace/Evidence MMDiT/execution
long launcher:          scripts/train_mainline.sh
RDT-8 launcher:         scripts/train_rdt_multitask.sh
smoke launchers:        scripts/smoke_mainline.sh / scripts/smoke_rdt_multitask.sh
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only)
config:                 configs/mainline/object_intent_dynamics_323.json
```

Release state: the source carries a conservative Schema28-core recovery overlay
under the Schema30 ABI, a profile-owned continuous-gripper codec boundary, and
an outlet-scoped CALVIN action repair. CALVIN's six arm values are already
relative TCP commands, so its two six-dimensional arm-field branches now both
encode that command directly and decode by the existing `0.75/0.25` blend;
neither branch is temporally differenced or integrated. Its explicit binary
command head remains isolated from the six compatibility-only future-gripper
coordinates, which are replaced by exact zeros before every dynamic model
consumer. These changes preserve the network blocks, parameter topology,
controller and complete 18-D output ABI. Focused tests, a real CUDA smoke, the
strict gripper-noise causal probe and read-only checkpoint replay passed. A
fresh CALVIN run from commit `f9cee96a` is active; no historical checkpoint was
resumed or migrated. These are interface observations, not an accepted
behavior release; live Pen, RDT and CALVIN process state remains in the rolling
handoff.

## Authority order

When documents disagree, use this order:

1. active source plus the supplied run's serialized `run_context.json`;
2. this contract;
3. current issues and repair plan;
4. the two current detailed references: RDT adaptation and R1/R2 closure index;
5. archived replay/design/log documents and Git history.

An old filename, log banner, launcher comment or conversation statement cannot
override manifest, source, config, dataset and checkpoint identity.

## Living-memory policy

This contract stores accepted architecture semantics, not a chronological
experiment diary. `CURRENT_MAINLINE_ISSUES.md` stores one current observation
per unresolved decision, and `ACTIVE_MAINLINE_HANDOFF.md` stores one current
operational snapshot. A new comparable audit replaces the older observation in
place; do not append epoch-by-epoch narration.

Archive a superseded observation only when it is required to explain an
accepted/rejected source change, a semantic-contract transition, a causal sign
reversal that affected a decision, or a baseline that remains necessary for
future comparisons. Git
history and the raw run directory are the default recovery mechanisms. Keep
only decision-making statistics, source references and reproducible commands
in repository memory; never copy raw JSONL rows or probe dumps into it.

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

### Training-only Teacher and training call graph

Future DINO supports enter only the no-grad Teacher and auxiliary targets.
Teacher builds once per training batch and zero times in deployment.

The active recovery training call has one ordinary online encode, one formal
velocity forward, and one loss composition:

```text
sample one noisy FlowMatchingState
  -> one formal velocity forward on the encoded cache
  -> compose action/future/auxiliary losses once
```

There is no train-time endpoint estimate and no train-time W rebuild. The
detached endpoint/self-conditioning lifecycle is retained only as historical
Schema30 evidence; the AMP/BF16 cache isolation fix from `d8a77a1` remains in
the code for any explicitly no-grad parameterized scope.

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

The deployment ABI also owns the frozen DINO boundary: RGB preprocessing,
encoder compute dtype, and the reference encoder batch shape are serialized and
validated. The current CALVIN cache uses bf16 with reference batch size 32;
online history rows are causally padded with the final observed row to reproduce
that CUDA kernel shape before slicing back to the three real history rows.

### Optional hybrid-v1 training and deployment (this branch)

The explicit Pen config
`configs/mainline/object_intent_dynamics_323_pen_hybrid_v1.json` selects schema
32. Schemas 30/31 keep their original training/sampling semantics and default
config digests. Hybrid is a new randomly initialized experiment, not a replay
or continuation of an arm-only checkpoint.

Hybrid keeps the 24x18 value/adjacent-difference field. The existing learned
arm-only B-spine provides cubic K=12 coarse/detail input beside the raw lift;
all six gripper fields retain raw input. The latest composite role boundary is
called on the actual clean outputs of each pass. It uses retained raw rows,
keeps motion logits as a typed endpoint sidecar, checks endpoint freshness and
leaves Pen gripper action authority with the continuous codec. It is never
inserted as a chart round-trip inside an ODE stage.

`runtime/hybrid.py` owns the same E5 proposal -> one differentiable W rebuild
-> H5 refined lifecycle for training and deployment. Each pass starts from the
same original physical noise. H5 means five Heun intervals / ten field reads;
both passes retain a separate endpoint read on the final corrected field.
Rollout field evaluation disables dropout without disabling autograd. Static
observation augmentation and pointwise-flow training retain their prior roles.

Training composes the existing pointwise flow/objective ledger plus supervision
on the actually refined rollout action. The extra decoded-action and gripper
trajectory terms reuse the existing .08/.03 budgets, multiplied by the explicit
hybrid rollout weight (1 in the selected config). Hold-before/no-event,
transition, and post-event persistence partition all gripper rows and each own
one third of the rollout gripper budget. Target masks select loss rows only;
they never enter the solver or gate execution. The proposal/W path is not
detached. The no-grad Teacher is still built once, and future observations or
actions cannot enter the rollout input.

The training forward count is one pointwise call plus 15 rollout physical
calls plus two endpoint calls. Non-reentrant activation checkpointing is
serialized and recomputes rollout activations in backward; that extra compute
is not hidden as an inference speedup. Deployment fastpath/compile/CUDA Graph
are outside this experiment. Fresh CUDA/BF16 parameter-owner, loss-ledger,
memory and checkpoint tests are required before the single formal Pen launch.

## Historical Schema30 semantic delta (not active in recovery)

The following source/config changes were present in the historical Schema30
checkout. They did not add a block, parameter,
buffer, optimizer group, loss weight, RNG draw or deployment pass:

1. `[reverted]` S sums complementary owners without the Schema28 fixed K/type
   mean and per-branch RMS contract.
2. `[reverted]` typed W innovations combine learned chronology with physical
   action; recovery restores the Schema28 action-carrier modulation.
3. `[reverted]` camera support was treated as metadata; recovery restores the
   Schema28 validity-times-support coordinate reduction.
4. `[superseded]` recovery initially restored current-qpos gripper anchors.
   The active non-core repair instead uses the profile-owned causal command
   boundary consistently in gripper encode/decode/loss/evaluation.
5. `[reverted]` the historical single public validity boundary is not the
   recovery owner layout; Schema28 typed validity/camera-carrier semantics are
   restored.
6. `[retained]` P3 optional `source_depths` uses the public
   cardinality-checked compiler path.
7. `[retained]` the unconsumed `proposal_condition_dropout` field is removed.

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
6. Physical validity and camera support are producer-owned. The Schema28
   camera-validity-times-support reduction is the named coordinate/transport
   consumer; confidence and allocation share are not substitutes for it.
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
11. Pen/RDT deployed gripper behavior comes from continuous physical
    value/delta branches. CALVIN uses the explicit binary command-state head
    and maps argmax to `{-1,+1}`; its legacy continuous gripper field remains
    compatibility/audit-only. In CALVIN binary mode the six trailing future
    gripper coordinates are replaced by exact zeros before every ODE-dependent
    dynamic consumer, while current gripper state remains available through
    observed state/action history. CALVIN's native arm rows are relative TCP
    commands, not absolute poses: both six-dimensional branches of the fixed
    18-D field encode the same command and decode as
    `0.75 * branch0 + 0.25 * branch1`; temporal differencing, cumulative decode
    and adjacent-command smoothness are forbidden for this outlet. Direct
    branch consistency remains active. Pen/RDT retain the historical absolute
    plus adjacent-delta arm chart exactly. Decoded events are evaluation
    metrics, never a runtime gate. For the continuous gripper codec, Pen/CALVIN
    use current action-state as the gripper boundary and RDT uses the previous
    executed command; this same boundary owns row-zero delta, horizon-wide
    `grip-anchor`, cumulative decode, loss, and evaluation.
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

## Modular cut-point contract

The accepted composition boundary for future source work is function-level,
not the current `top.py` / `restored_bottom.py` file boundary. Implementation
must land as one atomic source unit that moves the registered owners, rewires
the complete static/dynamic/training/runtime paths and installs component
selection together. A façade-only or half-migrated graph is not an accepted
intermediate architecture.

The logical composition graph is:

```text
OutletActionAdapter
  -> ConditioningStage
  -> ObservationStage
  -> SharedRoleQueryBridge
  -> GroundingStage
  -> IntentStage
  -> WorldStage
  -> P1Stage
  -> PolicyCompilerStage
  -> ControlledTransitionStage
  -> ExecutionBottomStage(TerminalActionController)
  -> OutletActionAdapter.finalize
```

The static and dynamic call planes are separate:

```text
one call per observation:
  conditioning
    -> observation/G1-G3
    -> ObjectFactSet
    -> S/coarse PhysicalActionCondition
    -> action-tagged CandidateWorld
    -> static FactualPrecisionDock
    -> ControlledTransitionSource

one call per ODE node:
  outlet-prepared physical field
    -> shared action query + V120SeedContext
    -> dynamic CompletedP1PolicyState
    -> P2/P3 ObjectPolicyPlanDeltaBank
    -> ControlledTransitionState
    -> V120 execution bottom
```

The target slots and their current function sources are:

| Logical slot | Current source functions | Required public result |
|---|---|---|
| `ConditioningStage` | `ClearVLAMainlinePolicy.encode_online` goal/history masks and `HistoryActionProposal.forward` | conditioned online input, proposal state and the two keep masks |
| `ObservationStage` | `RestoredV120ObservationCompiler.prepare` and `build_grounding_bank` | prepared V120 observation and lossless grounding bank |
| `SharedRoleQueryBridge` | `RestoredV120EvidenceBottom.sample_role_table`, `grounding_canvas`, `clean_action_basis_tokens` and `action_and_context` | one shared role sample, G canvas, clean basis, action query and one `V120SeedContext` |
| `GroundingStage` | observation `begin/advance/finalize_progressive_grounding`, `ObjectIntentDynamicsTop.run_progressive_grounding` and `grounder` | `ObservationEvidence`, `ObjectFactSet` and the exact completed G3 rollout |
| `IntentStage` | the intent and coarse-action portion of `ObjectIntentDynamicsTop.build_online_context` | `ObjectIntentState`, `CoarseActionIntentState` and canonical `PhysicalActionCondition` |
| `WorldStage` | `build_candidate_world` and `refine_deployment_world` | atomic `CandidateWorld(action_condition, dynamics)` |
| `P1Stage` | `LateRawDetailPolicyReader.forward` and `RestoredV120EvidenceBottom.complete_p1_fact` | static `FactualPrecisionDock` and dynamic `CompletedP1PolicyState` |
| `PolicyCompilerStage` | `ObjectIntentDynamicsTop.compile_policy` | `ObjectPolicyPlanDeltaBank` plus P2/P3 trace values |
| `ControlledTransitionStage` | `ControlledTransitionDynamics.build_source` and `forward` | separate cached source and per-node transition state |
| `ExecutionBottomStage` | `RestoredV120EvidenceBottom.forward` and `compile_evidence_view` | `BottomDecoderOutput`; no observation, world, Teacher or outlet identity input |
| `TerminalActionController` | `EvidenceLatentMMDiTActionDecoder._read_output_heads` and every direct terminal `velocity_head` candidate read | physical velocity, motion state and optional outlet command state |
| `OutletActionAdapter` | `PhysicalActionFieldCodec`, sampling finalization, outlet action terms and validation accumulation | canonical core field plus explicit deployed/native outlet result |
| `TrainingTargetStage` | `teacher_supports`, `build_training_targets` and history-proposal target construction | training-only detached Teacher and named target bundle |

These boundary rules are non-negotiable:

1. One experiment instantiates exactly one implementation per slot. Registries
   are lazy; they must not construct unused alternatives, consume their RNG or
   place inactive trainable heads in a `ModuleDict`.
2. The shared role table is sampled once per observation and reused by static G
   and every dynamic action-query call. The action query and
   `V120SeedContext` are produced atomically by the same bridge.
3. One completed G3 rollout tensor is passed by identity to static P1 and
   `ControlledTransitionSource`. It remains `[B,4*C*8*8,H]`; neither consumer
   may rebuild it from an object summary.
4. `ObservationEvidence` retains the N=49 and literal/detail candidates until
   static P1 completes. It may then be released from deployment cache, but it
   cannot be compressed at the module boundary.
5. `action_query`, `factual_base` and `policy_query_residual` stay separately
   named until `P2QueryDock.combined()`. Boundary containers pass references
   without `detach`, `clone` or a replacement projection.
6. `WorldStage` accepts only `ObjectWorldBelief` and
   `PhysicalActionCondition`. `CandidateWorld` keeps that exact condition
   atomically paired with the corresponding dynamics through P2 and outer
   refinement.
7. The dynamic bottom boundary accepts only the prepared physical field, time,
   shared action query, `ObjectPolicyPlanDeltaBank`, `V120SeedContext` and
   `ControlledTransitionState`. Goal, RGB/DINO, `ObjectFactSet`,
   `CandidateWorld`, Teacher and outlet/task identity do not cross it.
8. `TerminalActionController` is logically replaceable but physically injected
   into the V120 decoder because execution-candidate probes also consume its
   velocity reader. Replacing only the final endpoint head is not a valid slot
   implementation.
9. Outlet-native dimensionality, command alphabet, normalization, target
   encoding, finalization and validation stay in `OutletActionAdapter`.
   The shared core sees only the declared canonical seven-dimensional action
   and eighteen-dimensional physical field; outlet/task identity is not a
   hidden S/W/P condition.
10. Component boundaries never create a no-grad boundary. Teacher remains the
    only future-capable detached plane, and every online component retains
    ordinary end-to-end autograd.
11. Component selection, compatibility ABI and implementation names are
    serialized in the run context before component experiments can be compared.
    A source digest change still forbids exact resume unless that exact source
    identity or an explicit migration contract authorizes it.

The atomic rewrite, legacy-to-modular key map and behavior-equivalence gates are
maintained in `CURRENT_MAINLINE_REPAIR_PLAN.md`. The working source now uses
only the registered component hierarchy. The former `top.*`, `bottom.*`,
`action_codec.*`, `factual_reader.*` and `history_proposal.*` names exist only
as explicit checkpoint/optimizer mapping data; dual registration and runtime
compatibility façades are not part of the accepted graph.

Local closure evidence on 2026-09-04 is exact: the topology-complete reduced
dual-source gate compared 15,014 tensors / 10,443,735 shared elements with zero
difference across initialization/RNG, static and six-time dynamic boundaries,
raw and post-clip gradients, one optimizer step, sidecars and the two-pass
five-update deployment lifecycle. The combined mainline,
policy/runtime/structural, checkpoint/layout, AMP/data/interface/action-field,
RDT preparation and standalone B-spline suite passes 284 tests with three
environment-dependent tests skipped.
The final registered state inventory remains 1,391 keys with digest
`846b1edd7933b796882bcb5a8422816f768110fe9741282ba4435ac45927b7ca`.
The separately owned B-spine source is now integrated into the modular tree and
the local merge/identity gates below are closed. Real CUDA/BF16 and a real
read-only production-checkpoint replay remain release gates; no new training is
authorized from this source before those remote gates close.

### Provisional Schema31 B-spine candidate

Schema30 remains the accepted disabled-path baseline.  The opt-in Pen candidate
uses manifest schema 31 with digest
`a10eabe896acc214ea04338de7f85f568a4a9a13c0447212f96b15aecdf6edf6`,
component selection `execution_bottom=v120_evidence_mmdit_bspine0_v1`, and
config [`configs/mainline/object_intent_dynamics_323_pen_bspine0.json`](../../configs/mainline/object_intent_dynamics_323_pen_bspine0.json).
The resolved path-independent config digest is
`0a73c0a1f5c847227629fb7867834c8e1cba01ddf53bf4f3909fcaf16773fbfc`.
It is an experiment identity, not an accepted replacement for Schema30.

The only new calculation is a bottom-internal parallel numerical view of the
same deployed noisy physical field:

```text
u_raw   = unchanged NativeTimePhysicalActionTokenLift(x_t)
u_spine = BSpine0(x_t)
action += (u_raw + u_spine) * action_state_factor
```

`BSpine0` is fixed at `T=24`, cubic degree 3 and `K=12`. Its production basis
identity hashes the exact FP32 analysis/synthesis operators actually registered
by the bottom, excluding the unused backend-dependent lossless-detail QR chart;
that runtime-operator digest is
`f4d169cdeab9606dfacb92abbbc71bc3dbb7a4abefb8ef5244bc411670caab34`,
and its complete spec fingerprint is
`a2234eb6c9f553c47e793e11c8734d8cfadfbaaf86c5b950dab8f672965a8c10`.
Fixed analysis/synthesis run in FP32.  Independent zero-initialized, bias-free
coarse/detail maps preserve the five physical-field roles and add 18,432
trainable weights at hidden width 512; no knot, gain, normalizer, loss, clip,
top carrier, ODE step, W rebuild or output ABI changes.

The full Schema31 inventory is 168,435,611 parameters / 1,395 parameter
tensors, 152,064,880 trainable parameters / 1,073 trainable tensors, 1,403
state keys and 24 optimizer groups.  All ten new parameter tensors have the
single `bottom_spine` owner at the existing bottom-decoder `0.7x` LR and normal
decay. `spine_zero` is the evaluation-only model intervention and preserves
the learned computation for diagnostics while zeroing only its action-stream
contribution. Validation names its two uses separately:
`spine_zero_refined_pass` holds the learned-proposal W cache fixed, while
`spine_zero_full_lifecycle` applies it to proposal and refined passes so the
intervened proposal owns the single W rebuild. Both reuse the primary initial
physical noise; only the latter is complete deployment attribution.

The disabled Schema30 graph is still certified against the immutable
pre-modular capture at `atol=0`, `rtol=0`: 15,014 tensors / 10,443,735 values
and 206,237 metadata paths have zero differences. The candidate source-tree
digest recorded by the post-lifecycle report is
`ad226fd56bcc7ec3eb1d19e08791e095a5526866d00ef3f043cfa6275b350ba9`;
the checkout-newline-independent executable source-closure digest is
`ababb7ce0a87973de273e5d69bda94d91339ad217b9ccd658f54ecf68cc64ddc`.
Local B-spine gates cover fixed-basis rank/partition/endpoints, exact
coarse-plus-detail closure, independent raw/coarse/detail JVP/VJP, complete
loss-side owner gradients, CPU BF16, zero-init bit identity, matched learned
fixed-cache and full-lifecycle `spine_zero`, one formal decoder call, twelve
deployed calls, optimizer ownership and Schema31 checkpoint round trip with
Schema30 exact-resume
rejection.  Real Pen CUDA/BF16 VJP, smoke, runtime/memory and read-only
checkpoint replay remain mandatory before the single fresh Pen run starts.

### Feasibility audit (2026-09-04)

This decision was checked against the live CPU source, rather than inferred from
file names. A structural probe instantiated the baseline selection with seed 0,
then attached the same already-constructed child modules under the proposed
owners. The result was:

```text
baseline inventory:       168,417,179 parameters / 1,385 tensors
trainable / optimizer:    152,046,448 / 1,063 parameters
state keys / groups:      1,391 / 23
state-key digest:         70a8a5be21de40c460de6cff899942d5331837700db289350a0b1920c133b053
post-init RNG digest:     d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
legacy-key collisions:    0
mapped missing/unexpected: 0 / 0
mapped values/storage:    equal / shared
attachment RNG change:    none
module/parameter aliases: 0 / 0
```

The same probe on the CALVIN binary selection materialized only its command
head (1,389 parameter tensors, 1,395 state keys, 1,067 trainable parameters),
which confirms that outlet selection can remain lazy rather than retaining
inactive trainable alternatives. The current `encode_online` and `velocity`
functions are straight-line orchestrators; their component-call order is
stable and the deployment lifecycle test passes the expected two five-update
passes, one W rebuild and twelve bottom calls. The focused policy/runtime/
structural suite currently passes 149 tests on CPU. Reproduce it with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_mainline_policy.py tests/test_mainline_runtime.py tests/test_mainline_structural_contracts.py
```

This proves structural feasibility, not completed equivalence. Two implementation
gates were exposed and are mandatory in the atomic source unit:

1. A logical hierarchy changes the raw `model.parameters()` traversal order even
   when every mapped tensor is identical (the proposed hierarchy had 1,385
   positional differences). The rewrite must carry an explicit legacy parameter
   order for optimizer construction, global norm/clipping and any diagnostic
   reduction, or prove an exactly equivalent registered order. Matching only
   optimizer group membership is insufficient.
2. `EvidenceLatentMMDiTActionDecoder` still has seven direct candidate reads of
   `velocity_head` (probe, differentiable candidate, prefix/idle and both
   execution paths). They must all be routed through the injected terminal
   controller; moving only `_read_output_heads` would leave a hidden old exit.

The harness must also cover non-tensor sidecar state: execution warm-up/progress,
evaluation interventions and non-persistent query/position buffers are not
represented by the ordinary state-key map. Their setters, load behavior and
runtime values are part of the equivalence surface.

Therefore the answer is **feasible with a strict topology-preserving relocation**:
the source can be rearranged in one atomic unit while preserving the same
calculation DAG, operation order, tensor references, RNG draws and stateful
execution lifecycle. The behavior gate must additionally compare the explicit
parameter order, all seven terminal sites, raw gradients and the post-step
optimizer result before the new hierarchy becomes authoritative.

## Loss, gradient and optimizer ownership

The logged loss ledger is authoritative. Raw auxiliary magnitudes do not imply
optimization dominance; interpret `loss_contrib_*` and `loss_group_*` first.
The main groups are action, representation and execution. On continuous
Pen/RDT outlets, the retained `.03` gripper-trajectory budget supervises
continuous transition/persistence on the deployed codec branches; it does not
fund an event classifier. CALVIN's action-flow and decoded-action objectives
are arm-only and supervise both direct command branches; their consistency
term compares the branches, while the adjacent-command smooth-delta term is
exactly zero because it would be an unintended acceleration target. CALVIN
removes all continuous-gripper terms from the backward ledger and uses the
explicit command CE at weight `.1`. When both command states occur in one
batch, CE rows are reweighted so their exact horizon-weighted mass is equal
while preserving the configured mean objective budget; a single-class batch
remains unmodified.

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

### Pen/RDT gripper boundary decision

The compared Schema29 Pen and RDT-8 runs both came from source
`d8a77a19cfbd7520ae790b3938e2d1fb3a8a7a6f`. The logged value
`0.873665452` is RDT-8 epoch 4, task `grab_stick_into_bottle`, metric
`validation_gripper_rmse_physical`; it is a 64x24 aggregate in the producer's
source-native chart, not a state value or verified SI physical unit. Its
normalized counterpart is `0.514941`.

At epoch 4, Pen/RDT-8 gripper RMSE was `0.222692 / 0.336409` normalized and
`0.147643 / 0.570762` source-native. The raw ratio `3.865825x` decomposes into
normalizer scale `2.559052x` and normalized difficulty `1.510647x`. Data probes
show Pen action gripper equals qpos gripper, while RDT action is a command whose
qpos response has a different scale and about two control steps of lag. On the
same probed rows, gripper-field RMS is `1.096491 / 1.149281` (train/val) with a
qpos anchor, `1.051814 / 1.102704` when only row zero is repaired, and
`0.645774 / 0.658893` when the previous-command boundary owns all 24 rows.
Therefore partial row-zero rewriting is forbidden: encode, decode, loss, and
evaluation must share the same profile-owned boundary.

### Physical chart metadata

`clearvla.data.physical_chart` records unit-bearing nominal references beside
the numeric action/state profile (`action_unit`/`state_unit`, nominal absolute
limit/span, and optional mechanical lower/upper bounds). It is metadata only:
it does not clip, normalize, decode, condition the model or enter the existing
profile digest.
For the Pen chart, the six arm channels use `rad` with
`nominal_abs_limit=pi`; their symmetric `[-pi, pi]` range therefore has a
`full_scale=2*pi`. The gripper uses `rad` with a nominal `[0, 100 deg]`
reference (`1.745329... rad`), so its absolute limit and span are both that
value. These are nominal references, not a claim that every arm joint has the
same mechanical limit. RDT and CALVIN channels remain explicitly
`source_native` with unknown limits until source-side units are verified; the
RDT qpos-to-command scale is a chart conversion, not a physical full scale.
`ArrayNormalizer.minimum/maximum` remain observed train-split extrema, and the
legacy gripper probe's `angle_max_deg` is diagnostic input; neither silently
defines a physical full-scale contract.

## Identity and checkpoint contract

- The manifest, resolved config, executable-source digest, dataset inventory,
  normalizers, language artifact, optimizer ownership and RNG state are
  serialized in `run_context.json`/checkpoint metadata.
- Branch and run-directory names are descriptive only. The current branch name
  still contains `schema29`; the manifest remains Schema30 with the recovery
  ABI suffix.
- Pre-recovery Schema29/Schema30 checkpoints and current-qpos-anchored RDT
  checkpoints are rejected for exact resume/deployment by the component and
  deployment ABI. RDT must start a fresh checkpoint after this repair. Smoke
  checkpoints are gate artifacts, not formal initialization sources.
- `validate_mainline_checkpoint.sh` is read-only: optimizer, scheduler and RNG
  load are disabled and no checkpoint is written.
- Formal output directories must be new and empty. Checkpoint writes are atomic;
  do not overwrite an existing run to continue a different identity.

## Historical release evidence (pre-recovery)

The following gates passed for the historical Schema30 checkout before the
local recovery overlay; they do not certify the dirty working tree:

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
completed behavior anchor. The recovery graph has since started fresh and has
four complete epochs, but remains a midpoint candidate until E8.

## Run and audit

Canonical commands and the current remote environment are maintained in
[`clearvla/mainline/README.md`](../../clearvla/mainline/README.md) and the live
handoff. Formal runs use the XVLA Python environment on the server; non-
interactive SSH must make that environment visible in `PATH`.

Hard stops are: non-finite values, lineage/identity failure, an open loss
ledger, formal parameter-owner VJP disappearance, checkpoint ABI violation or
memory above the 22 GiB release boundary.

Finite gradient/preclip threshold crossings are secondary telemetry, not a
release gate. Their count, maximum, owner or clustering does not independently
stop a run, rank a version or authorize clipping, normalization, gain or
loss-weight changes. Retain them for retrospective correlation and escalate to
a targeted numerical investigation only when the same event is reproducibly
coupled to sustained failure to recover, optimizer/parameter damage, validation
regression or another hard-stop failure. Judge and stop on that demonstrated
failure, not on the `spike` label itself. Early event F1, small geometry RMS or
capacity warmup likewise does not independently stop a run.

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
