# Current ClearVLA architecture contract

Updated: 2026-09-22 UTC

This is the compact source of truth for the active mainline graph. Read it
before changing the V96+ top representation, Flow-DINO/JEPA, role hierarchy,
language/history conditioning, long-horizon phase handling or the
top-to-bottom evidence path.

Operational state belongs in
[auxiliary/ACTIVE_MAINLINE_HANDOFF.md](auxiliary/ACTIVE_MAINLINE_HANDOFF.md),
open behavior questions in
[CURRENT_MAINLINE_ISSUES.md](CURRENT_MAINLINE_ISSUES.md), and future repository
or refactor work in
[CURRENT_MAINLINE_REPAIR_PLAN.md](CURRENT_MAINLINE_REPAIR_PLAN.md).
Expanded historical narratives remain recoverable from Git and the archive;
they do not define the current graph.

**Scope:** the current workspace implementation and resolved mainline configs;
this is not a declaration that every experiment runs this graph or that the
dirty workspace is a published release. For a checkpoint, first identify its
pinned source, serialized config, deployment ABI and run context. Historical
CALVIN direct-arm runs do not acquire this contract's shared-codec or S repairs
by being evaluated today. Schema, branch name and documentation date alone do
not identify a trained model.

For a first reading, use [Agent quick contract](#agent-quick-contract),
[Default and opt-in selections](#default-and-opt-in-selections),
[Active graph](#active-graph) and
[Non-negotiable invariants](#non-negotiable-invariants). Outlet, migration,
residual-RL and B-spine sections are references for those explicit selections.

## Agent quick contract

~~~text
capability:             object_intent_dynamics_323
default manifest:       schema 30
optional B-spine:       schema 31 only when explicitly selected
registered layout:      clearvla_mainline layout 2
component selection:    mainline-modular-v1
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global objects:         K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0; two learned adjacent flows
training:               one online encode, one formal velocity pass, one loss composition
deployment:             proposal ODE, one W rebuild, refined ODE
shared action core:     24 x 18 value/adjacent-difference/gripper field
default config:         configs/mainline/object_intent_dynamics_323.json
Pen launcher:           scripts/train_mainline.sh
RDT-8 launcher:         scripts/train_rdt_multitask.sh
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only)
working branch label:   codex/schema29-mainline (historical name, not semantics)
~~~

The default is the Schema28-core recovery behavior under the schema-30 ABI,
with a profile-owned continuous-gripper boundary and outlet-scoped binary
adaptations. The shared action codec, G/S/W/P graph, training call count and
two-pass deployment lifecycle are common across outlets. CALVIN and the
repaired ManiSkill v2 base change only their outlet sampler/chart and terminal
binary-command behavior; the ManiSkill choice is explicit and does not change
Pen/RDT/LIBERO defaults. The opt-in B-spine is an experiment selection, not a
new default.

## Default and opt-in selections

These are source/config selections, not a table of trained-model successes.
Source admission, structural checks, completed training and behavior validation
are separate evidence states; their current records belong in the issue ledger
and dated handoff.

| Boundary | Default in this workspace | Explicit alternative |
|---|---|---|
| Manifest / bottom | Schema30, B-spine disabled | Schema31 only for a selected B-spine component |
| S interval object query | `goal_history` | `history_only` is a single-edge research control |
| W camera condition | `motion_prior_only` | `coordinate_role_v1`; separate old-checkpoint initialization migration |
| W action condition | `interval_mean_v1`, four interval rows | `sequence_prefix_v1`, 24 source rows; separate initialization migration |
| Target identity / P2 spatial intent | `post_pool_only`; typed S selects interval after spatial pooling | `shared_target_prior_v1` changes only P2 addressing; `target_action_bottleneck_v1` binds target identity into one supervised physical proposal before goal-free downstream physical correction |
| Visual NPY read transport | `mmap` | `pread`; existing cache layout and logical data contract are unchanged |
| ODE schedule | Uniform E5 proposal/refined | Registered Q5 plus its versioned flow-step context |
| Residual RL | Absent from the mainline trainer | External frozen-base pilot in `clearvla.rl` |

An opt-in's presence or passing source tests does not promote it to the default.
The checkpoint's own selection remains authoritative for evaluation and resume.

## Authority and document ownership

When records disagree, use:

1. for a run, its pinned source, checkpoint identity and serialized run context;
   for a source change, the actual workspace source and resolved config;
2. this contract;
3. the current issue ledger and repair plan;
4. the bounded current adapter/design document named by this contract;
5. archived documents and Git history.

Branch names, run-directory names, old launchers and conversation labels are
descriptive only. They cannot override source, config, manifest, dataset or
checkpoint identity.

Maintain this file as architecture, not chronology. Replace decisions in
place. Keep only accepted semantics, invariants, typed boundaries and stable
source locations. Metrics and unresolved causal questions belong in the issue
ledger; PID, host, path and next-command state belong in the handoff; completed
proof belongs in Git or the archive. Never copy checkpoints, caches, raw logs
or full probe dumps into repository memory.

## Active graph

### Observation and grounding

~~~text
RGB/DINO at -8,-4,0
  -> V120 raw/DINO compiler + two learned adjacent flows
  -> current-only G1/G2/G3 progressive grounding
     G2 rematerializes N=49 fine candidates once
  -> camera x 8x8 x local-M hypotheses
  -> dense global K=4 plus null grounder
  -> ObjectFactSet and reversible K-to-chart correspondence
~~~

G cannot read language, executed-history proposal, noisy action or future
Teacher evidence. Local-M rows are hypotheses, not persistent objects. The
dense grounder owns real/null mass; G3 may refine only the distribution over
real K objects. Reconstruction uses detached current DINO on observed cells
and exports one K-specific content value shared by reconstruction, S, W and
the detached Teacher.

The paired A/B observation controls are opt-in and keep the default graph
unchanged. A disables only the online current-context mask while preserving
the encoder's random-draw position; B additionally selects
`canonical_rgb_lattice_v1`. In that contract, outer RGB pixel centers are the
canonical frame and every raw, descriptor, DINO patch/pooled, flow, support,
transport, Teacher-G and P1 coordinate crosses a named `ChartSpec` before it
is consumed. The executable DINO processor is center-crop geometry, and its
resize/crop/patch shape, backend and chart digest are admitted through the
deployment ABI. Per-source visibility/interpolation support remains separate;
it cannot be replaced by a shared DINO validity mask. The legacy normalized
chart remains the default and is bitwise-preserved for old checkpoints.

K is an equivariant representation axis, not a fixed color or instance ID.
The reversible chart correspondence is a typed interface; an entity-to-K
mapping in a particular observation must be verified before using it as an
oracle intervention or a target-grounding score.

### Intent, action condition and world

~~~text
T5 + observed state/history + ObjectFactSet
  -> S public interval carrier + typed [interval,K,type] relevance
  -> typed-free CoarseAction proposal
     default: [B,4,7] interval means
     opt-in sequence_prefix_v1: [B,24,7] physical action rows
  -> outlet-owned physical action condition
     default PhysicalActionCondition: four values + adjacent deltas
     opt-in PhysicalActionSequenceCondition: 24 source/value/delta rows,
       current boundary, control-step time, chart and normalizer identity
  -> W(ObjectWorldBelief, selected physical action condition)
     W1 owns intervals 0/1; W2 reads W1 and owns intervals 2/3
  -> action-tagged CandidateWorld / FutureObjectDynamics
~~~

S is the sole intent owner. W cannot read goal tokens, S values, coarse hidden
tokens, Teacher or noisy ODE action. Semantic successors retain
[B,4,K,D]; transport, covariance and camera support retain
[B,4,K,C,*] until P2. W predicts no validity, visibility or status authority.

The accepted default W camera condition remains
`top.world_camera_condition_mode=motion_prior_only`.  The opt-in
`coordinate_role_v1` research unit adds each producer-valid camera's current
image-plane coordinate and its declared `data.camera_names` role to the
existing motion-prior condition.  A shared zero-initialized projection
modulates only the existing geometry carrier, so its initialization is the
exact default function, an exact-zero carrier stays zero, and invalid
object/camera rows stay zero.  Role codes are keyed by the declared names, not
hard-coded array indices; jointly permuting camera data and the declared role
order is equivariant.  These inputs are view conditions, not calibrated
extrinsics or world coordinates.  Enabling the mode adds a parameter and is
therefore an explicit component initialization from an old checkpoint, never
an exact resume.  The only admitted old-checkpoint path is
`--init-checkpoint ... --init-model-contract-migration
world_camera_coordinate_role_v1`: it retains every old model tensor, requires
the same camera-role order/data/normalizers/language, initializes only the new
weight to exact zero and starts fresh optimizer/schedule/RNG state.  It does
not change S, P2, the coarse proposal, objectives or the one-rebuild deployment
lifecycle.  Its source-drift allow-list is exactly `config.py`, `dynamics.py`,
`policy.py`, `top.py`, `runtime/checkpoints.py` and `train.py`; P2
`compiler.py`/`transition.py` drift is rejected rather than inherited from a
generic initialization allowance.

TargetFact currently exports its physical camera mass in sorted canonical
camera-name order, while the live ObjectFactSet/W/P2 charts retain the
serialized data-camera order. Until those shared C-axis owners carry one
explicit role permutation, target-action construction therefore requires
`data.camera_names == tuple(sorted(data.camera_names))` and fails closed for a
non-canonical order. The formal CALVIN order is `("top", "wrist")`; this is
an admission guard, not a claim that arbitrary camera-role permutations are
already supported.

The accepted action condition remains
`top.world_action_condition_mode=interval_mean_v1`.  The opt-in
`sequence_prefix_v1` unit closes the deterministic temporal blind spot of the
four clipped means without changing W's four prediction intervals, P2, the
bottom or the bounded two-pass lifecycle.  CoarseAction produces and is
supervised on the same 24 normalized physical rows used at deployment.  One
independent `PhysicalActionSequenceCondition` carries the complete outlet-
native source sequence, deterministic canonical value/delta views, current
boundary and contiguous control-step times; it never impersonates the legacy
four-row type.  For relative-command outlets the arm value is the prefix sum
of zero-centered commands and the arm delta is the command itself.  This is a
command chart, not measured TCP pose or SE(3) integration; no extra `dt` is
applied.  Gripper values are not accumulated.  Absolute Pen/RDT charts retain
absolute value plus adjacent difference.  Normalizer offset/scale metadata and
its fingerprint remain FP32 and are independent of the action/autocast dtype;
only a temporary arithmetic view is cast to the source dtype.  A BF16/FP16
condition therefore cannot silently acquire a different normalizer identity.

W reuses the existing physical row projection, then applies an order-sensitive
causal recurrent encoder.  Its four conditions read rows 1--8, 1--16, 1--24
and 1--24 respectively; the last condition means prediction beyond the known
24-row control prefix, not invented controls for rows 25--48.  In this mode W1
keeps the shared typed common factual and action-independent so the second near
prefix cannot flow backward into the first.  The CandidateWorld identity owns
the full sequence/time/boundary/chart container.  Initial coarse and deployed
rebuild conditions use the same OutletAdapter factory; a binary deployment
command is first put back into the checkpoint normalizer chart, while the
online coarse gripper estimate remains differentiable.  Online inference and
training caches bind the configured condition schema/profile, and every real
OutletAdapter consumption boundary revalidates mode, concrete condition type,
profile, chart, arm dimension, prefix, and normalizer identity even when
diagnostics are disabled.

Enabling this unit adds exactly one zero-initialized 24-row coarse-query offset,
one time projection and one recurrent cell.  The old four-query basis,
H-to-action head and physical row projection are retained.  Old checkpoints
may enter only through `--init-model-contract-migration
world_action_sequence_prefix_v1`; camera mode, S/P2/bottom, data, language,
normalizers, objectives, runtime and Q5 identity must otherwise match, and a
fresh optimizer/schedule/RNG is mandatory.  This migration is separate from
the camera-condition migration.  Its source-drift allow-list covers only the
files needed to introduce this action-condition unit; P2 `compiler.py` drift is
rejected.  Structural admission proves row identity, prefix causality and
train/deploy ABI consistency only; action benefit still requires matched
training and autonomous closed-loop validation.

In the default/post-pool graph, the language/object repair is part of this
existing S path, not a CALVIN sidecar or a second object container. S keeps two
explicit K-preserving views:
the content-only interval evidence remains the owner of the established typed
relevance decomposition, while a bounded symmetric
semantic/appearance/geometry-enriched public K memory supplies the coarse
object read. Both views are masked by producer-owned validity. Goal/history
innovations condition the interval object query, and that same conditioned
query is carried into the coarse object's K read and history read. Thus an
instruction can change the K-preserving object attention and typed S context,
while W receives intent only through the outlet-owned physical action
condition. P2 has a separate direct S path: it reads typed S with the already
formed spatial posteriors and uses the selected typed/public S context for
interval selection. This remains the default
`top.p2_spatial_intent_mode=post_pool_only` behavior.

The opt-in `shared_target_prior_v1` adds one S-owned FP32 `[B,I,K]` address
field before P2 spatial pooling. It is computed from the same typed semantic,
appearance and geometry scores before validity amplitude multiplication, by a
bias-free exact-zero 3-to-1 owner whose construction consumes no random draw.
The address is centered over producer-supported K after first subtracting one
legal reference K, so uniform evidence is exact zero under FP32 and autocast.
The same K field is added once to the semantic K logit and once to the geometry
K*C logit, broadcasting only over C. It cannot create support, rewrite W
semantic/transport values, select a separate target for the two readers or
change the interval terminal directly. Supported non-finite address values are
rejected; unsupported K values are quarantined to exact zero. The retained
post-pooling typed/public context owns only interval selection, so it is not a
second target-K decision.

No raw goal tokens, color labels or rank-1 selected-object tensor are added to
W or the bottom. Enabling the opt-in adds exactly
`intent.organizer.target_object_address.weight` with shape `[1,3]`; old
checkpoints enter only by explicit component initialization. The standalone
`p2_shared_target_prior_v1` migration admits only that model unit. The explicit
`p2_shared_target_prior_pread_v1` combination additionally changes the physical
NPY reader from default `mmap` to `pread`, while retaining identical dataset,
language, normalizer, component and model-state identities apart from the one
new exact-zero parameter. Both start fresh optimizer/schedule/RNG state and are
never exact resume across modes.

The CALVIN raw-overlay reader is also admitted in that pread migration's
source allow-list. Its frame-chart discovery is endpoint-validated and lazy:
the `ep_start_end_ids.npy` start/end indices are the finite evidence for a
candidate prefix/width/suffix, unrelated indexed archives are ignored, and a
partially covered chart fails closed. This avoids an eager full-directory glob
without weakening raw-frame ABI validation.

When a converted HDF5 loader receives a non-empty manifest of flat,
root-relative episode identities, the same pread migration admits an exact
`*.hdf5`/`*.h5` path fast path. It validates every identity, resolves only the
requested files in sorted identity order, and fails closed on a missing file;
recursive patterns or nested identities retain the legacy discovery path. This
changes discovery cost only and does not change episode parsing, ordering,
duplicate-identity checks, or the model/data ABI.

The explicit `target_action_bottleneck_v1` identifier names a full
single-target factorization. It replaces the legacy language-conditioned
interval-K reader, the coarse-action raw-K read, and P2's semantic
action-to-K query with one observation-level TargetFact owner. Protected goal
plus observable history query the enriched
content/semantic/appearance/geometry K memory once under producer support and
emit one finite FP32 posterior `p_target[B,K]`. This posterior is converted
immediately into declared physical statistics: per canonical camera,
coordinate first/second moments, transport-prior first moment and readable
mass, plus object-readable, existence and unresolved mass. Identity and
physical readability remain separate typed quantities:
`m_object(K)=p_target(K)*validity(K)` and
`m_camera(K,C)=p_target(K)*camera_validity(K,C)` are carried in FP32 without
renormalization; the C axis is admitted only under the canonical camera-order
guard above. A learned
bias-free projection turns only these statistics into `target_summary`; raw
content/appearance/RGB and raw K rows do not reach the full-goal action
proposer. The exact-zero score head starts with a legal uniform posterior;
only the derived log-prior innovation is centred.

Target mode deliberately registers neither S `scene_read/scene_pool` nor the
coarse `object_read`. The 24-row coarse proposal A0 reads the operation
carrier, observable history and the already materialized target physical
summary; after that boundary it cannot reopen K or use `p_target` as a second
selector. W remains goal/S-free and sees target intent only through the
outlet-owned A0 physical action condition while still modelling every
producer-valid object. The same `p_target` owns every later target-object
marginal.

P1 receives `p_target`, the exact same ObjectFactSet/G3 owner and the aligned
24-row A0. Its target factual address and nonlinear local K/C refinement are
A0-free. Cameras are fused inside each K and K is contracted under
`p_target` exactly once; only this already-selected target fact is then
modulated by the operation/A0 condition. The fourth lane is a separate
goal-free, A0-conditioned physical coverage read and carries no target
future-transport value. A structural 3+1 block mask forbids target/coverage
cross-read. The P1 dock binds the exact G3 chart tensor owners, not merely a
same-shaped chart or address token; joint K relabeling remains legal, while
foreign-observation and local-M mixing are rejected.

P2 has two explicit roles and no additional learned K selector. Identity keys
and spatial addresses use `p_target` (and `p_target(K)q(C|K)` for geometry), so
low readability cannot silently become a new object selector. Action-facing
semantic/geometry values, typed context and effective support use the declared
physical masses `m_object`/`m_camera`; they attenuate unreadable evidence but
are never renormalized. A fractional validity change can therefore reduce a
target value to zero while leaving its identity key unchanged and its support
closed. For producer support
`s(K)` with `n=sum s`, the non-target scene role is the fixed complement
`r(K)=s(K)(1-p_target(K))/(n-1)` when `n>=2`, and exact zero for `n<2`.
Physical readability is not multiplied into `r` or renormalized across K, and
missing consumer evidence removes its contribution without renormalizing mass
onto another K. This is an expectation over the other objects under one
uncertain primary target, not a multi-target membership model.

Target and scene keep separate spatial keys, factual contexts, support and
physical-I posteriors. They share the terminal parameters and interval
definition, not posterior mass: a target with no readable camera cannot erase
a readable non-target consequence. The two scene value projections start exact
zero and are trained only by ordinary task loss. Their output remains a named
scene effect through an independent RMS boundary and consequence/P3 interface;
it enters the execution bottom as `p3_scene_consequence` and never rewrites the
target effect, protected target consequence, temporal lane or state-change
lane. Compatibility fields such as `target_object_address_logit` and interval
attention remain derived views of `p_target`, not trainable selectors.
Unsupported values are quarantined before learned projections and coordinate/
covariance arithmetic; the selected-evidence admission rejects supported
non-finite values even when diagnostics are disabled. No raw goal, color label,
hard contact/phase rule, or direct language side channel enters W or the
execution bottom.

There is no target-binding teacher label or auxiliary target loss. Ordinary
coarse/action/representation losses train the selector through the consumed
P1, W, P2 and terminal paths. `target_score`, the two P2 scene value
projections and the public CT physical readout start exact zero; target
query/key projections and other declared physical projections retain ordinary
initialization. The score head and zero-start terminal owners receive direct
ordinary task-loss VJP before their upstream bases are expected to open.
These staged openings are required training diagnostics, not separate
objectives, gains or entropy quotas.

The legacy-to-target migration adds thirteen declared states for
`interval_mean_v1` or twelve when the source already owns the sequence row
offset. It retires sixteen states from `post_pool_only` and seventeen from
`shared_target_prior_v1`, including both raw-K readers, legacy typed relevance,
legacy P1 goal/history selectors and P2's second K query. Old checkpoints may
enter only through
`--init-model-contract-migration target_action_bottleneck_v1`, with identical
data, language, normalizer, camera/action ABI, objective and runtime semantics
apart from this graph selector, and a fresh optimizer/schedule/RNG. Both
source modes are checked by exact state-set and source allow-list; undeclared
state or source drift is rejected.

An already-trained target-action checkpoint enters the narrowed public-CT
system only through
`--init-model-contract-migration target_action_system_v2`. This migration
adds exactly `transition.physical_delta_head.weight` and
`execution_bottom.transition_delta_lift.weight`, requires the prior target
mode with otherwise identical config/data/runtime identity, rejects a source
whose trained camera-column order was not already canonical, and starts a
fresh optimizer/schedule/RNG. The physical readout starts exact zero; the
evidence lift retains its ordinary finite initialization. The formal CALVIN
candidate uses `pread`, `coordinate_role_v1`, `interval_mean_v1` and Q5; it
does not silently combine the separately tested sequence-W alternative.

The default `top.interval_object_query_mode=goal_history` preserves this graph.
An explicitly serialized `history_only` research control removes only the goal
innovation from the interval K query. History, the public carrier's goal input,
the coarse conditioned reads, parameters, objectives and action codec remain
unchanged. It is a single-edge ablation, not a no-language baseline or a new
default; checkpoint/deployment graph identity retains the selected mode.

### Precision, policy, transition and execution

~~~text
completed G chart
  -> one cached P1 high-resolution read
     24 factual queries, N=49, real 3x3 RGB/detail microgrid
  -> FactualPrecisionDock

noisy action + time + factual detail
  -> dynamic P1 residual
  -> P2 semantic-K and geometry-K*C selection per interval
  -> no-null physical interval terminal per type
  -> semantic + geometry consequence

P2 consequence + S/action context
  -> optional temporal P3
observed state change + S/action context
  -> optional state-change P3

completed G3 rollout
  -> cached ControlledTransitionSource
noisy action + consequence + dynamic P1
  -> legacy: ControlledTransitionState
  -> target mode: private 512-row centered CT value
                  + (action query, A0, current state) x private G3 address
                  -> PhysicalTransitionInnovation D[B,24,18]
all protected/optional carriers + target-mode lifted D
  -> V120 execution bottom
  -> terminal physical velocity + target-mode D exactly once
  -> motion and optional outlet command
~~~

Protected consequence and raw dynamic-P1 precision are no-null carriers. Only
the temporal and state-change P3 lanes own zero-null choices. Evidence MMDiT,
CVAE/workspace, continuous capacity and execution-value machinery remain part
of the bottom.

In `target_action_bottleneck_v1`, the completed 512-row G3 selector, private
centered CT value and coefficient tensors never cross the CT owner boundary.
CT uses a 24-row query derived from the noisy action representation, A0 and
current state to address that private chart, then one bias-free exact-zero
projection emits only `PhysicalTransitionInnovation.delta_v[B,24,18]`. The
same physical D has two explicit consumers: a learned 18-to-hidden lift gives
the existing evidence machinery a 24-row transition memory, and
`TerminalActionController` adds D exactly once to every complete Q5 velocity
read, including dynamic candidates, deployment fastpath, fixed candidates and
diagnostics. Batch-row selection uses `index_select`; B-by-N candidate
expansion preserves sample ownership. Continuous outlets admit all 18
channels. Binary outlets zero compatibility-only channels 12:18 at the outlet
boundary, and both the normal policy path and direct bottom seam fail closed
if an unmasked or non-finite D arrives. Legacy modes retain their original
`ControlledTransitionState` path exactly.

In the default P2 reader, the per-interval semantic K and geometry K*C spatial
posteriors are formed from the post-P1 action query, W keys, geometry address
and producer-owned support; direct typed-S values are then read using those
posteriors and condition the four-interval terminal. In the opt-in reader, the
shared S target-K field conditions those two spatial posteriors first, while
the same post-pooling typed context continues to condition only the interval
terminal. In target-action mode, the one S-owned `p_target` fixes both target
semantic K and target geometry K mass before P2; action/coordinate evidence
may select C within that K but cannot move mass to another object. A fixed
complement supplies a separately named non-target scene role. After A0, raw
language and raw K cannot reopen target identity. W's per-camera transport
values are camera-image-plane 2-vectors;
P2 pools them over K*C and interval before its shared 2-to-hidden geometry
projection. Neither this pooling nor W camera conditions imply calibrated
world-frame geometry. These are graph contracts, not a claim that the trained
target selection or geometry use is behaviorally adequate; attribution and
closed-loop acceptance belong in the current issue ledger.

## Training, deployment and runtime

Future DINO, action and state are available only to the no-grad Teacher and
named training targets. Teacher builds once per training batch and never in
deployment.

The default training path is:

~~~text
encode online observation once
  -> build detached Teacher/targets
  -> sample one FlowMatchingState
  -> one formal velocity forward
  -> one loss composition
~~~

There is no train-time endpoint estimate or train-time W rebuild.

Deployment and validation perform exactly two complete five-update ODE passes
from identical initial physical noise:

~~~text
W(coarse)
  -> proposal ODE at t = 0,.2,.4,.6,.8
  -> decode 24-row proposal
  -> rebuild W once from the selected physical action condition
  -> refined ODE at t = 0,.2,.4,.6,.8
  -> final action
~~~

This is one bounded correction, not a fixed point. The final action may differ
from the action that conditioned the rebuilt W, so final interval/delta
mismatch remains a residual metric. Recomputing W after the final action
without another policy consumer would not close anything.

The default remains the exact uniform E5 grid above. The registered Q5
schedule (`t_i=(i/5)^1.25`) is opt-in. A non-uniform schedule now carries the
versioned bottom-context contract
`FlowStepContext=(t,Δt,k/N,endpoint)` through the sampler and the matching
training bridge, so the learned bottom field can account for the interval it
is asked to integrate. The schedule schema/fingerprint records that context
contract; an old sampler-only schedule artifact is rejected rather than being
treated as an exact-resume alias. The bottom still predicts an instantaneous
physical velocity and the integrator applies `Δt` exactly once. The 24-row
action-time chart, B-spline basis and W rebuild remain separate contracts.

Observation/G/S, static P1 and ControlledTransitionSource build once per
observation. Dynamic P1, P2/P3, transition and the bottom run at every ODE
node. Only the bottom receives the numerical `FlowStepContext`; it is not a
semantic or outlet/task condition. The t=1 read supplies retained endpoint
motion/command state but does not update action. Execution candidates remain
mandatory for train/eval loss forwards and disabled during ordinary deployment
passes.

The targeted CALVIN evaluator records the explicit local replan cursor
`calvin_replan_cursor_v1`: each planned chunk carries its cumulative
environment-step origin and each executed row carries both that origin and its
local row. The evaluator uses the same chunk-retention/observe state machine
for `execute_rows=1`, 8 and 24; this fixes provenance and row execution
semantics only. It does not claim that the current policy consumes an absolute
global phase, which remains a separate model/ABI question.

An explicit deployment-only fastpath may reuse the exact block-major dwell
prefixes and the current decision's already-computed neutral dwell-one result.
It retains the complete candidate chart, values, probabilities, diagnostics,
terminal baseline and action ABI; it is disabled by default, forbidden in
training, and retains no reuse state across a decision or ODE node.

The deployment ABI owns DINO preprocessing, dtype and reference batch shape.
When the serialized profile requires a reference batch, causal padding may
reproduce that encoder shape before slicing back to the real three history
rows.

## Action and outlet contract

The shared canonical action is seven-dimensional and its physical flow field
is [B,24,18]: six arm values, six arm adjacent differences and six continuous
gripper coordinates. The two arm branches and cumulative decode remain common
to every outlet.

| Outlet | Native boundary | Mainline claim |
|---|---|---|
| Pen | Seven-dimensional continuous action; current action-state owns gripper anchoring | Core-behavior outlet |
| RDT-8 | Right-arm seven-dimensional projection, high plus right-wrist RGB; previous executed command owns gripper anchoring | Adapter and cross-task outlet, not native three-camera/depth/14-D bimanual support |
| CALVIN | Seven-dimensional relative TCP command; current action-state plus explicit binary gripper command | Outlet-specific command semantics, not a replacement shared codec |
| LIBERO | Seven-dimensional normalized OSC_POSE relative command plus continuous gripper; previous clipped command owns the gripper boundary | Isolated external comparison; its ingress/egress adapter does not alter the shared codec or network core |
| ManiSkill v1/v2 continuous | Seven-dimensional normalized `pd_ee_delta_pose` relative command; previous clipped command owns the gripper boundary | Retained compatibility baseline |
| ManiSkill v2 binary | Same seven-dimensional relative arm command; private native gripper command owns dimension 7 and emits strict `{-1,+1}` | New StackCube base outlet; does not modify the shared 18-D codec |

For every continuous gripper profile, one causal boundary owns row-zero delta,
the full-horizon anchor, cumulative decode, loss and evaluation. Event masks
select rows only; decoded events are evaluation metrics, never runtime gates.

CALVIN applies exactly three outlet rules:

1. motion sampling scores normalized command minus normalized raw zero;
2. before W, four centered relative commands become cumulative canonical
   displacement plus per-interval command delta;
3. the binary command head maps argmax to {-1,+1}, while the six
   compatibility-only future-gripper coordinates are zero before every dynamic
   ODE consumer.

Its dataset boundary is independently fail-closed. Split membership is by
official `ep_start_end_ids.npy` source trajectory, never by language annotation
row or sampled window. Every real action through the annotation's terminal
state minus one must occur in a 24-row policy target. The extra support needed
by the 48-frame Teacher follows CALVIN's absorbing convention: repeat terminal
RGB/state, zero relative arm commands and hold the final binary gripper command.
Synthetic absorbing rows are excluded from normalizer fitting. These are
outlet data-interface rules and do not alter the shared policy graph. An
annotation is admitted only when its source trajectory also provides the full
24-frame causal history; short-prefix annotations are rejected before HDF5
materialization rather than failing after a partial output is written.

Pen and RDT bypass the CALVIN W adapter numerically. Task identity may control
sampling, validation and logging but is not a hidden S/W/P condition.

LIBERO has its own fail-closed ingress/egress contract.  Conversion derives the
policy instruction from the official filename-based `Task.language` surface and
records a BDDL `:language` value, when present, only as simulator provenance.
The official BDDL task inventory is mandatory conversion input; HDF5 language
metadata alone cannot assert the evaluator instruction identity.
The HDF5 action/state arrays are required to be native seven-dimensional
`OSC_POSE`/EEF charts with a reset-zero then previous-action `action_state`.
The upstream LIBERO `create_dataset.py` records `obs[t]` after executing
`action[t]`; that post-action row must never be presented as the observation
that generated the same action.  The causal boundary converter therefore
renders row zero from raw `states[0]`, shifts released post-action rows right by
one, and treats the released final post-action row as the genuine terminal
observation.  Prefix and terminal-suffix roots retain a numeric-only copy of
the legacy real state rows solely to keep the E8 state-normalizer fingerprint
identical; that reference is not model evidence.
The controlled LIBERO continuation pair is explicit: `causal_prefix_v1`
admits centers `0..T-49`, while
`causal_prefix_terminal_suffix_v2` admits `0..T-1` after appending exactly 48
absorbing rows (zero arm, held final gripper, repeated terminal state/RGB).
Both reserve `1 prefix + 7 strict` or `1 prefix + 1 tail + 6 strict` in B8,
keep strict validation as the checkpoint-selection metric, and allow the
strict-to-causal transition only through the tested model-only migration
`libero_window_boundary_supervision_v1`.
An optional retarget training overlay may augment only the terminal-suffix
train split.  Each admitted source demonstration owns one symmetric nonzero
XY translation pair generated by simulator replay: the object initial free
joint and the expert EEF trajectory change together, while the original
receptacle remains the release target.  Manifest lineage, HDF5 content hashes,
success, tracking/saturation/placement bounds and paired response direction
all fail closed before loading.  The immutable base train split alone fits the
action/state normalizers; overlay rows are appended only after that fit, and
base validation/test membership and strict checkpoint-selection metrics remain
unchanged.  This changes dataset/cache identity but not the model, objective,
codec or outlet ABI.  A terminal-suffix checkpoint may initialize such a run
only through the tested model-only migration
`libero_retarget_training_overlay_v1`, which requires byte-identical action
and state normalizer SHA-256 values.  Its source allow-list composes the
ordinary model-only initialization/replay allow-list with the data-boundary
files needed by the overlay; selecting the data migration must not revoke an
already-admitted validation-only source change, and it still rejects drift in
every source file outside that union.
Conversion writes into a staging root, audits the complete inventory, and
atomically publishes only an audited root.  XML model assets are relocated
from the released `chiliocosm` absolute prefix to the installed LIBERO asset
root with an existence check; the relocation is serialized as provenance.
Terminal replay starts at raw `states[0]` and executes the complete recorded
action sequence, because the OSC controller target is hidden outside the
flattened MuJoCo state.  The raw writer's final reward/done entries are
official terminal markers (the upstream writer sets them on the final row),
so replay reward/done/success is retained as an audit comparison and never
replaces or silently filters the raw label; mismatches are reported per
episode.  Conversion still fails closed on malformed arrays, missing assets,
state-restore failure, or a replay process exception.  Evaluation applies the
fixed init
state, performs the official five zero-action physics warmup without reading
success/done, then asks the policy for a 24-row chunk and executes its first
row after normalized `[-1,1]` clipping.  The clipped row is the next
`action_state`; success is evaluator-only.  These are LIBERO outlet rules and
do not add a second network forward path or change the shared action codec.
The evaluator resume identity also includes the action/state normalizer
digests, so a relocated bridge is allowed but a silently re-charted bridge is
not.  Fixed init states retain their official first-N identity, and every
resumed rollout must carry a complete executed-action audit.  `libero_100` is
an archive union rather than an evaluator suite, so 90 and 10 are evaluated
separately.  Only a checkpoint-backed, all-task 20-episode/600-action run with
the five-step warmup, seed 10000 and 128-pixel camera input is labeled as using
the official protocol; transport smokes never acquire that label.  Result
identity includes `image_side` plus content digests of each used first-N fixed
state matrix and BDDL file.  Bridge endpoint is relocatable provenance, not
policy or benchmark identity.  Partial per-task checkpoints remain explicitly
incomplete until the final bridge-identity check succeeds.

Physical-chart units and nominal limits are metadata. They do not clip,
normalize, decode, condition the model or silently define a full scale.
Observed normalizer extrema and diagnostic probe constants are not physical
unit declarations.

RDT data and adapter detail lives in
[auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md](auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md).
Simulation and benchmark operational detail lives in
[../development/SIMULATION.md](../development/SIMULATION.md).

## Optional simulator residual-RL pilot

The opt-in `clearvla.rl` pilot freezes a verified same-task ClearVLA/DINO base
and learns an external bounded unit-residual SAC actor with independent twin
Q critics. It runs after original outlet finalization and native decoding,
before environment clipping/execution; the actual executed command owns the
next history. Reward/success stay outside deployed policy inputs. The mainline
18-D codec, G/S/W/P, one-W-rebuild two-pass sampler, Teacher and supervised
trainer are unchanged; execution-value heads are not RL critics.

New base preparation selects `maniskill_pd_ee_delta_pose_7d_v2` with the
`maniskill_binary_command` / `maniskill_7d_binary_v2` outlet. The legacy v1
profile remains readable for existing runs, but is not admitted as a binary
base by the repaired RL runner. Passing `--gripper-output-mode continuous`
remains an explicit compatibility baseline; it does not silently convert an
existing checkpoint. Both are explicit StackCube/panda_wristcam boundaries,
not Pen/CALVIN/LIBERO aliases. Relative-command W conversion, previous-command
gripper anchoring, the shared 18-D codec and the neural graph remain unchanged.

The ManiSkill binary outlet follows the Calvin isolation contract without
copying Calvin's rejected direct-arm chart: a private two-class head owns the
native gripper command, all six compatibility future-gripper field coordinates
are zeroed before every dynamic consumer, and finalization plus W rebuild write
the strict native command (`+1` open, `-1` close). Binary preparation admits
only exact `{-1,+1}` values in both `action` and `action_state`; it therefore
requires a fresh data/config/checkpoint identity rather than an exact resume of
a continuous v2 run.

V2 records a causal rotation vector of the fixed chart `Rx(pi)^-1 R_tcp`,
choosing the nearest equivalent log branch from the last observed rotation
and clearing that state on reset. This is an invertible outlet-side chart
with the same 7D state width, not a globally nonsingular SO(3) parameterization.
Training and deployment use exactly the same producer implementation.
Every original source action owns a center from 0 through source_length-1;
at least 48 real post-success hold steps provide future Teacher support.
Only pre-reset history may pad: repeat the reset RGB/state and use the safe
reset action_state for unexecuted command slots, never the first future action.
All future supports remain actual recorded steps. Episode attributes and
admission checks enforce first-executed-row coverage, not merely presence
somewhere in a 24-row target. New data/caches/normalizers/checkpoints have new
identities; no live v1 directory is recharted or exact-resumed into v2.
Environment and reset-seed provenance remain serialized at the outlet.
During fixed source-plus-hold collection only, a successful task terminal may
be continued with live physics; the native terminal is retained in evaluator
metadata. Failure/time-limit termination and ordinary online-RL termination
are never suppressed by this data-collection rule.

Adapter checkpoints have separate base/source/environment/replay identities.
This is an implemented experimental interface, not a validated RL improvement
or a replacement mainline. Preparation, limitations and gates live in
[`../../clearvla/rl/README.md`](../../clearvla/rl/README.md).

The single-task information sampler carries fractional event quotas across
batches (B4/default alternates 0/1 dedicated event rows); integral B8 quotas
retain their historical sample sequence. Logging separates configured fractions
from realized dedicated quota. ManiSkill open/close names follow positive/negative
native gripper changes respectively; this naming correction does not alter total
event F1, RMSE, or losses. These changes require a fresh source identity.

## Optional B-spine contract

The default execution-bottom selection has B-spine disabled. Opt-in selections
use dedicated Pen configs and distinct schema-31 component identities. Each
reads the same deployed noisy physical field after the unchanged codec and
adds a parallel fixed coarse/detail numerical view beside the raw lift:

~~~text
u_raw   = existing noisy lift of x_t
u_spine = fixed-analysis/coarse-detail lift of x_t
action += (u_raw + u_spine) * action_state_factor
~~~

The retained `fixed_bspline_coarse_detail_v1` comparison applies the shared
cubic K=12 temporal split to all 18 physical fields. The preferred isolated
candidate `fixed_bspline_arm_only_v1` applies it only to the arm-absolute and
arm-delta groups (`11000` in the serialized action-group order); all six
gripper fields reach the bottom only through the unchanged raw lift. The two
implementations have distinct manifest bottom identities and component
selections even though both remain in the schema-31 experiment family.

The follow-up candidate `fixed_bspline_arm_coarse_context_v1` keeps the same
arm-only fixed chart and `11000` mask but owns only the learned coarse
projection `P(X_arm)W_c`; it has no learned detail lift or gripper input. The
raw action lift remains the complete row-local path, so this candidate tests
only the non-local arm context supplied by the spline. It is a fresh Schema31
identity and is not a migration or a new default.

The matched routing control `fixed_bspline_arm_private_reader_v1` uses that
same coarse arm chart and mask, but keeps the chart out of the shared MMDiT
action seed:

~~~text
raw physical field -> existing raw lift -> shared MMDiT -> terminal heads
coarse arm chart  -> ArmPrivateBSpineReader -> arm-only physical correction
~~~

`ArmPrivateBSpineReader` is a non-affine LayerNorm followed by a bias-free
projection to the first `2*arm_dim` physical channels. The coarse lifts remain
zero-initialized (and reader construction restores the host RNG), so model
construction and the initial output are baseline-preserving while the arm loss
can open the new route. The correction is applied by
`TerminalActionController` to every candidate and final physical read; it is
never read by the shared MMDiT blocks, gripper state/command head, motion head,
or gripper physical channels. This is an explicit arm-private comparison, not
a hidden second codec or an arm/gripper stop-gradient claim.

No B-spine experiment may change the codec, target, source noise, flow
objective, loss weights, ODE nodes, W rebuild count or output ABI. The raw
lift remains active. Component choice, action-group mask, basis digest and spec
fingerprint are serialized in run context. Acceptance requires matched
behavior evidence; schema 31 does not promote any implementation to the
default. For the private-reader control, report the spline-owner and
private-reader gradient probes separately so a shared optimizer-owner total
cannot hide an arm/gripper routing difference.

The canonical basis mathematics and standalone representation boundary live in
[../../clearvla/action_representations/bspline/README.md](../../clearvla/action_representations/bspline/README.md).
The first coarse/detail run's bounded early evidence is retained in
[auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md](auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md).
It is a historical snapshot; current candidate status belongs in the issue
ledger and dated handoff, scoped to the selected component and checkpoint.

## Non-negotiable invariants

1. Camera, spatial, local-M, global-K, N=49, interval, horizon, basis and type
   axes stay real until a named consumer. Reduced evidence cannot be recreated
   with expand and called original evidence.
2. Online evidence keeps ordinary autograd unless this contract names a
   no-grad Teacher or audit scope. Activation gradients do not substitute for
   parameter-owner VJP.
3. Learned flow is a continuous source-relative prior, never a forced-nonzero
   target or route quota.
4. S owns intent. It cannot manufacture W support/value or enter W through a
   hidden second path.
5. W is the only future-world producer. ControlledTransition consumes policy
   transition evidence and cannot create another world. In target-action mode
   its G3/value/coefficients are private and its only public result is one
   finite `[B,24,18]` physical innovation.
6. Physical validity and camera support are producer-owned. Confidence or
   allocation share cannot silently replace them. Every S/coarse/W K read
   receives that support; invalid rows are quarantined before normalization
   and object-axis reduction, and an all-invalid row has a finite zero-output
   fallback. Camera-role conditions must use the serialized declared role
   order, remain jointly equivariant with that camera axis and cannot invent
   value behind zero support.
7. P1 retains N=49 and its real 3x3 detail read until factual selection is
   complete.
8. In compatibility readers, semantic K and geometry K*C selection remain
   independent and complementary. In target-action mode they share exactly one
   target-K posterior, retain separate value/camera/interval reads, and cannot
   learn a second target selector. Physical interval selection has no learned
   null.
9. Neutral P2 is algebraically neutral: zero effect and interaction leave the
   protected factual consequence unchanged.
10. The V120 seed, terminal contracts, CVAE/workspace, Evidence MMDiT,
    capacity and execution paths remain present.
11. The shared 18-D value/adjacent-difference action field and cumulative
    decode cannot be replaced for one outlet. Outlet-native conversions stay
    at the adapter/finalizer boundary.
12. Future observation/action/state can affect detached Teacher or targets
    only; changing it cannot change deployed action.
13. Every trainable parameter has exactly one optimizer owner. Decoder-local
    clipping precedes global clipping; finite post-clip values cannot hide a
    missing raw owner gradient.
14. Formal runs fail closed on missing language mapping, identity mismatch,
    stale CandidateWorld identity and non-finite values.
15. Checkpoints, tensor caches, raw logs and full probe dumps never enter
    architecture-memory documents.
16. Numerical step conditioning and semantic object binding are independent
    contracts. `FlowStepContext` may enter the bottom flow, while raw language,
    color/object pointers and role logits may not; object binding must be
    resolved upstream and cross the seam only as a compiled physical condition.
17. In target-action mode, A0 is the language-to-physics boundary. P1 may use
    A0 only after target factual K/C refinement and the one `p_target`
    contraction; W/P2/P3/CT/bottom cannot reopen raw goal-conditioned K.
18. A target-mode CT physical innovation is applied directly by every terminal
    read exactly once and may enter hidden evidence only through its declared
    lift. Binary outlet masking precedes both uses; no raw private CT chart may
    re-enter the bottom.

## Typed boundary summary

| Boundary | Required semantics |
|---|---|
| OnlinePolicyInput | Causal RGB/DINO, state, executed history and language only |
| FutureSupervision | Disjoint training-only future action/state/DINO evidence |
| ObjectFactSet | K=4 physical objects plus explicit null and observable object/camera mass |
| ActionIntentDock | Public S interval/history/K memory plus producer-owned K validity mask; no typed fact re-entry |
| PhysicalActionCondition | Default four physical interval means plus current-anchored deltas |
| PhysicalActionSequenceCondition | Opt-in complete 24-row source/value/delta condition with boundary, control-step time and chart identity; never a four-row compatibility view |
| ObjectWorldBelief | Compact current G belief; no S/Teacher/noisy action |
| CandidateWorld | One exact action condition atomically paired with FutureObjectDynamics |
| FutureObjectDynamics | Semantic successor/delta and camera-resolved transport/covariance |
| FactualPrecisionDock | Completed protected P1 detail; no new reader or compression |
| CompletedP1PolicyState | Static factual base separate from dynamic action/time residual |
| SelectedIntervalEvidence | Interval-retaining semantic/geometry values with no-null physical terminal |
| ControlledTransitionSource | Exact completed G3 rollout, built once per observation |
| PhysicalTransitionInnovation | Target-mode finite `[B,24,18]` CT correction only; no selector, private value or coefficient surface |

## Registered component hierarchy

One experiment selects exactly one lazy implementation for every slot:

~~~text
ConditioningStage
  -> ObservationStage
  -> SharedRoleQueryBridge
  -> GroundingStage
  -> IntentStage
  -> WorldStage
  -> P1Stage
  -> PolicyCompilerStage
  -> ControlledTransitionStage
  -> ExecutionBottomStage(TerminalActionController)
  -> OutletAdapter
~~~

Static observation work and per-ODE dynamic work are separate. Boundary
containers pass existing references without detach, clone, hidden projection
or reconstructed axes. WorldStage accepts only ObjectWorldBelief and
the configured `PhysicalActionCondition` or
`PhysicalActionSequenceCondition`. Language-to-object binding is owned above this seam:
S/P2 may use language with the language-free ObjectFactSet and must compile the
result into the existing physical action/consequence carriers. The dynamic
bottom accepts only the prepared physical field, flow time, bottom-local
FlowStepContext, shared action query, compiled physical plan, V120 seed and
the mode-specific transition boundary; goal, RGB/DINO, ObjectFactSet, raw
object pointers, private G3/CT value, color labels and task identity cannot
cross it. A physicalized consequence or `PhysicalTransitionInnovation` may
cross because it is an execution condition, not a second semantic binding
route.

TerminalActionController owns every physical-velocity candidate read, not just
the final head. OutletAdapter owns native dimensionality, normalizer-aware
chart conversion, W projection, dynamic sanitization, finalization and metrics;
it may coordinate these formulas but cannot leak outlet identity into the
shared graph.

Component selection and compatibility ABI are serialized before comparison.
Legacy layout/key/order translation is checkpoint compatibility data; it is not
a second registered owner or a runtime façade. Component boundaries never
create a no-grad plane.

## Loss, optimizer and diagnostics

The logged loss ledger is authoritative. Interpret weighted contributions and
group totals before raw auxiliary magnitudes. The main groups are action,
representation and execution.

Pen/RDT continuous-gripper trajectory terms supervise the deployed continuous
codec. CALVIN action-flow and decoded-action objectives are arm-only and its
binary command uses ordinary horizon-weighted cross entropy by default. An
explicit CALVIN-only `motion_event_v1` experiment may reallocate that same
horizon supervision mass across rows using zero-centered relative-TCP motion
and a bounded gripper-transition neighbourhood. It drops no rows, normalizes
the combined row weights back to the original horizon mass, is serialized in
the resolved objective config, and leaves Pen/RDT on exact unit frame weights.
Diagnostics and matched interventions are audit-only unless source shows a
positive objective weight.

Every train window requires an exact ledger, finite raw owner gradients and
one optimizer owner per parameter. Any compatibility ordering needed for an
old checkpoint is explicit and tested; constructor history is not an
architectural semantic.

## Identity and checkpoint contract

- Manifest, resolved config, component selection, source provenance, dataset
  inventory, normalizers, language artifact, optimizer ownership and random
  continuation state are serialized in run context/checkpoint metadata.
- The current dataset inventory is composite and must not be called a raw-file
  checksum; identity separation remains an open repair-plan item.
- Exact resume fails closed on incompatible architecture, component, data,
  optimizer or continuation state. A migration requires an explicit tested
  contract.
- Validation replay is read-only: it does not load optimizer/scheduler/random
  continuation state and writes no checkpoint.
- Formal output directories are new and empty, and checkpoint writes are
  atomic. Filesystem overwrite protection is separate from architecture
  compatibility.
- A smoke checkpoint is gate evidence, not a formal initialization source.

## Release and audit rules

Interface closure requires focused tests, state/owner coverage, real
CUDA/BF16 parameter-owner VJP, checkpoint round trip and bounded memory.
Learned-behavior closure additionally requires a complete comparable curve and
the matched intervention named by the current issue. Outlet claims require
their native metric and, where applicable, a closed-loop benchmark.

Hard stops are non-finite values, identity/lineage failure, an open loss
ledger, vanished formal parameter-owner VJP, checkpoint ABI violation or
process memory above 22 GiB. Finite gradient-threshold crossings, early event
F1, small geometry amplitude and capacity warm-up are telemetry unless tied
reproducibly to a hard failure or behavior regression.

## Authoritative source map

~~~text
identity/config:
  clearvla/mainline/manifest.py
  clearvla/mainline/config.py
  clearvla/mainline/interfaces.py
  clearvla/mainline/model/component_contracts.py

composition:
  clearvla/mainline/model/policy.py
  clearvla/mainline/model/components.py

observation/G:
  clearvla/mainline/model/restored_observation.py
  clearvla/mainline/model/observation_contract.py
  clearvla/mainline/model/grounding.py

S/W/P:
  clearvla/mainline/model/intent.py
  clearvla/mainline/model/dynamics.py
  clearvla/mainline/model/v120_p1.py
  clearvla/mainline/model/compiler.py

action/transition/bottom:
  clearvla/mainline/model/action_codec.py
  clearvla/mainline/model/transition.py
  clearvla/mainline/model/restored_bottom.py

training/runtime:
  clearvla/mainline/training/
  clearvla/mainline/runtime/
  clearvla/mainline/train.py
~~~

Historical replay provenance is indexed by
[archive/replay/README.md](archive/replay/README.md). Open it only for ancestry,
an old log or the reason behind a past repair; never reconstruct the current
graph from the ledger.
