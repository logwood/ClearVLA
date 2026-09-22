# ClearVLA structural rebuild work plan

Updated: 2026-09-22

This branch is an implementation of the user's authorized, sequential
structural rebuild. The immutable base is
`0f07160d692ec8c8302880420a3d93d74512c39b`; the read-only experiment control is
`a0e1f5d5b9736d37a51faec36039144f181d4b12`. Work stays on
`codex/structural-rebuild-20260921`. No master update, branch deletion, old
checkpoint migration, server-job change, or formal training is implied.
The older consolidation plan is recoverable at the base commit. It is not a
veto on the new authorization.

## Workflow

Establish semantics first; then review top-to-bottom, carrying every changed
producer interface through its consumers, supervision, masks and lifecycle.
At each unit, inspect forward information flow, reverse gradient ownership and
state/cache updates. A consumer-interface migration is **not** completion of
that consumer's deeper structural review. Revisit upstream decisions when a
later consumer exposes missing information. Do not freeze implementation
choices just to preserve old golden outputs.

## M6a saved-source boundary

An explicit action-matched supervision branch is now implemented on repaired
M5b. It uses labelled 8/16/32/48-step control prefixes, detached source masks,
shared W weights and separate training-only types/loss routing. The online
24-row action graph is unchanged. Targeted tests and source checks are recorded
per checkpoint; full M6 acceptance is pending. Do not omit the remaining
online far-control mismatch, physical interaction and entity-association work
merely because this subunit is saved.

## Status and dependencies

| Unit | Status | Completion boundary / next obligation |
|---|---|---|
| M0 source baseline | Implemented | Full base archive verified against Git blobs; final tree diff of the two experiment heads recorded; isolated branch and read-only source audit exist. Baseline test and diagnostic scope is in the handoff. |
| M1a real history time | Implemented candidate; validation recorded per commit | Shared source clock; separate state/control encoding; padding/dropout masks through S, coarse and proposal; bounded online history; new config/components/ABI. This is not all of M1. |
| M1b real-tail labels | Implemented candidate; validation recorded per commit | All labelled real current centers and independent action/state/visual support through loader, Teacher, auxiliary/formal losses and validation. Source files/caches and old defaults unchanged. |
| M1c native/feature state | Implemented candidate; validation recorded per commit | One CALVIN rotation-column encoder for current/history/future/online; native statistics and seven-dimensional commands remain separate; explicit new ABI and checkpoint checks. |
| M1d visual source time | Implemented candidate; verification recorded per commit | Source gaps, repeated frames, semantic/raw flow loss support, temporal/global G reads and future queries, S/W rates and Teacher duration use one observable clock. No claim about unprovided external controllers. |
| M2 observation/G1 | Implemented candidate; verification recorded per commit | Full FP32 G1 support in actual G2 candidate/value materialization; G3 observed-content quadrature and real per-candidate local P1 microgrids, explicit config/ABI. Deeper G2/G3 entity ownership remains M3/M4. |
| M3 G2 local observation | Implemented candidate; verification recorded per commit | One soft FP32 location law across typed attributes, full-support G2 evidence and joint G3 local-M refinement; actual binder/Teacher/P1 consumers adapted. Global entities and temporal reassociation remain M4. |
| M4a G3 context -> entity binder | Implemented candidate; verification recorded per commit | Completed current G3 sampled on actual support reaches real K competition; observed target/support authority unchanged; source-owned gradient and restore tests. |
| M4b current-image entity support | Implemented candidate; verification recorded per source | Complete spatial law in actual reverse lookup/reconstruction and one geometric summary; stable log-domain conditioning; query/image identities separated. |
| M4c1 causal observed entity evidence | Implemented candidate; verification per source | Existing inverse flows compose observed past onto current pixels; actual full-candidate evidence reaches K competition. Same finite window in training and online, no deployment-only state. |
| M4c2 remaining global/persistent identity | Open | Physical objectness, cross-window association and any matching recurrent training/online state; do not mistake finite-window grouping for persistent K identity. |
| M4d entity-motion chart | Implemented candidate; validation per source | Shared current-image spatial law for position and motion; includes earlier-source status correction. No extra parameters or world calls; view-aware downstream reasoning remains open. |
| M5a S/shared target | Implemented; exact-source acceptance recorded in artifacts | One soft K-plus-null binding consumed by real S/coarse/P1/P2, post-read mass, current spatial evidence, registered optimizer/checkpoint owners. Not task progress or persistent physical identity. |
| M5b S causal instruction reference | Implemented candidate; acceptance per exact source | Actual labelled origin in dataset, immutable runtime anchor, same-query per-view reference/current comparison consumed by S and real policy. No phase timer or persistent K-ID. Desired-relation/recognizer objectives remain for subsequent S/W supervision review. |
| M6 W1/W2 | Not started | Match supervised action to observed outcome; separate candidate prediction from demonstration-target training; explicit robot-object relations and known control horizon. |
| M7 P1/P2 | M5a target marginal/current-evidence seam implemented | Deeper target-local precision, view-aware future geometry and physical consequence semantics remain open; shared binding alone does not close them. |
| M8 P3 | Not started | Task-related local coordination and actual-prefix execution feedback; no task-clock or oracle state machine. |
| M9 transition/bottom/outlets | Padding values quarantined only | Audit actual consumers, redundant/frozen paths, action codec, clipping/history feedback and endpoint heads. Do not treat retained topology as approved. |
| M10 training/deployment/resume | M1a/history and M1b/label interfaces integrated only | Production CPU train/restore/sample and source-label masks are exercised; W supervision, endpoint context and new entity/task state lifecycle remain open. |
| M11 full release review | Not started | Rewalk output-to-input gradients and input-to-output semantics, remove temporary transport/compatibility paths, record all remaining unsupported claims. |

## M1a source and lifecycle map

| Producer | Consumer | Semantics / gradient / lifespan |
|---|---|---|
| `history_clock.py` | Dataset and online `HistorySnapshot` | Integer source offsets and Boolean provenance; never learned; one physical observation. |
| Loader / checkpoint policy | `ObservableHistory.timing` | Mandatory in new mode; no future target field; strict admission before online encoding. |
| Conditioning | S/proposal plus existing downstream consumers | Quarantine padding before projections; dropout changes action mask consistently; original immutable input is not modified. |
| TimedHistoryEncoder | S interval/readout and coarse | 11 time-ordered state/control slots; real-gap state rates; causal key masking; losses update real producer modules. |
| HistoryActionProposal | Cached executed memory / auxiliary proposal loss | Real sparse times; masked aggregation; absent memory is zero; learned future prior remains separately named. |
| Snapshot cache | Every dynamic/ODE consumer | No clock increment, mask mutation, or entity/task-state update within a numerical solver call. |
| Config/component selection | Model factory / checkpoint / deployment ABI | Explicit candidate identity; no automatic old-checkpoint or optimizer migration. |

## M1b real-label producer/consumer map

| Producer | Consumers reviewed in this unit | Invariant |
|---|---|---|
| Real terminal + labelled start | `future_clock`, window admission, sampler, audit progress | A stored suffix is not observed future; every admitted label owns a real row-zero command. |
| Dataset masks | Cached transport, actual loader, shared `FutureLabelSupport` | Keys can repeat terminal storage; independent masks must survive collation/device moves. |
| Future support | Teacher, recognizer, W/S targets, coarse/proposal losses | No partial fixed interval, unknown future attention key, or artificial hold label. |
| Action support | Flow bridge, action/decoded/command/motion and execution-value objectives | Missing rows remain source noise; no contribution or gradient from unavailable label payload. |
| Metric support | All target-dependent validation errors, event/motion counts, ablation errors | Denominator is real labels; report band support; keep model-only action deltas separately defined. |
| New config/source identity | Ordinary save/exact reload | No M1a exact-resume alias or old-weight migration exception. |

## M1c source and consumer map

| Producer | Consumers | Boundary |
|---|---|---|
| Native source profile and normalizer | Shared `data/state_features.py` encoder | Statistics retain original seven-dimensional physical state domain. |
| Feature encoder | Dataset current/history/future and online policy | Ten-dimensional observed state, same transform and rotation convention; missing future payload remains zero. |
| Explicit config | Component factory and restored visual/bottom configuration | State width may differ from command width only under a declared compatible chart. |
| State feature metadata | Deployment ABI and normalizer loader | Type-sensitive exact metadata; feature width 10, native normalizer/action width 7. |
| Source-owned tests | Actual dataset, loader, model, optimizer, checkpoint and sampling | External image/token transport may be a fixture; neural and state-conversion paths are production code. |

## M1d source and consumer map

| Producer | Consumers | Boundary |
|---|---|---|
| Source history offsets | Observation preparation and raw encoder | Strict ordered causal clock; duplicated pixels/masks share latest real copy before projection. |
| `VisualSourceTime` | Semantic/raw flow, dense temporal organizer, future queries, G blocks | Same known frame/pair support across objectives, keys, values, reductions and backpropagation. |
| Prepared visual packet | Grounding bank -> local facts -> global object facts -> world belief | Retain actual latest-pair duration as metadata; K permutations preserve it. |
| Observable displacement and duration | S/W feature rates and Teacher geometric prior | Rates and displacements are separate values. Missing observed motion is not proof of stationary physics. |
| New config | Factory/core bridge/checkpoint/deployment ABI | No silent fixed-gap fallback, fixed-FPS assumption or old checkpoint migration. |

The production candidate continues to use the real current image at reset.
A source-time support mask must never become a global visibility/success gate.
M2 now routes the full G1 parent measure into the real G2 sampler. Its
consumer migration retains explicit camera/candidate axes through G3 observed
content and P1 local patch quadrature, with mode-owned count and checkpoint
validation. M3/M4 must not revert these paths to a barycenter-only read while
unifying local attribute ownership and global/persistent entity identity.

## M2 source and consumer map

| Producer | Consumer | Contract |
|---|---|---|
| FP32 complete G1 logits | G2 real candidate sampler | Same camera support; no top-k or mean-location substitution. |
| G2 bounded correction | Every retained candidate coordinate | Correct points individually; retain actual historical source query. |
| Parent law + local evidence | G2 typed candidate and local-hypothesis reads | Integrate normalized parent probability, not candidate-count-dependent energy. |
| Geometry candidate law | G3 observed DINO content | Read the content at support points before expectation; tiled exact value bandwidth. |
| Actual candidate coordinates/RGB/detail | P1 real local microgrid | Local per-hypothesis 3x3 patches, masks and logit-space posterior conditioning. |
| Explicit candidate-support selection | Factory, validation, deployment ABI | No silent N=49 assumptions, opaque fallback or old-checkpoint alias. |

## M3 source and consumer map

| Producer | Consumer | Contract |
|---|---|---|
| Three G2 typed compatibility queries | Joint local observation law | Attributes may differ, but their candidate location posterior is shared and soft. |
| Full G1 parent + source validity | Candidate and local-M integration | No candidate-count reward; unsupported values cannot create mass or gradients. |
| One G2 location law | G3 canonical typed values and observed DINO content | Same entity-hypothesis location, distinct typed values; retain M2 current-camera support. |
| Three G3 residual heads | One refined M law | No independent reassignment of property identity; no real/null authority transfer. |
| Local-M law | Dense K binder, Teacher's current references, P1 | One source prior; downstream deeper review is not waived by interface compatibility. |
| Explicit local mode | Factory/state/deployment ABI/checkpoint | No silent fallback or old-checkpoint migration. |

M4 still needs to review global objectness and persistent association. A soft
mixture with aligned marginals may remain ambiguous; do not claim that M3 has
resolved every constituent into a physical object. More hypotheses or a hard
argmax are not automatic remedies. Keep complete observed support available
while designing the global entity state and its training/online lifecycle.

## M4a source and consumer map

| Producer | Consumer | Contract |
|---|---|---|
| Completed current third G block | Current-support expectation | Every retained candidate, geometry-owned location measure; no barycenter substitution. |
| Source-owned local context | Dense chart -> actual K competition | Mask before projection, no ignored feature or extra object container. |
| Observed raw/current DINO | Reconstruction and Teacher references | Remains independent target content; updated assignments are not physical identity truth. |
| Entity-only test objective | Third G block, context projection, binder | Verifies the new path without routing gradients through P1/decoder. |
| Config and deployment identity | Model factory, save/exact reload | Explicit opt-in, no old migration or persistent state masquerading as compatibility. |

M4a closes a concrete missing current-context edge, not the entire G3 redesign.
M4b retains ambiguous support through the current-image write. M4c must still
decide how entity identity is associated across actual observations. Any stateful association needs source-time
reset semantics and matched training burn-in/unrolling; do not add a deployment-
only recurrent cache or silently assume the same K index denotes the same entity.
The current-only reconstruction objective is not proof of physical segmentation.

## Review and publishing gates

Run `bash scripts/check_structural_rebuild.sh /path/outside/repository` in the
project-compatible Python/PyTorch environment. It executes the actual
production modules through source-owned tests, not a replacement toy policy.
It runs each selected test file in a fresh process to bound allocator lifetime,
keeps the complete per-file inventory/log/JUnit, and marks a killed, timed-out or
empty child as a failure. It records the untouched baseline separately and
refuses new Pyright errors;
Ruff must pass on all changed Python files. Historical repository diagnostics
remain visible in JSON. This differential gate does not claim the whole old
repository is type-clean. Tests that encode old optional-field assumptions may
be made mode-aware without dropping the underlying tensor assertion.

Publish a complete semantic unit (producer, consumers, supervision, tests and
docs) by normal fast-forward only. Verify the exact resulting source tree and
remote HEAD. Temporary transport tooling must be removed from the published
source. Do not publish raw videos, checkpoints, private logs, caches or secrets.
CI verification is separate from local verification; neither is formal robot
training or closed-loop task success.

## Design principles, not fixed algorithms

Preserve G/S/W/P responsibilities, causal online/target isolation, explicit
coordinate/action semantics and a recoverable old baseline. Candidate counts,
association algorithms, world intervals, memory sizes and solver call budgets
remain reviewable decisions. A new module or renamed type is not evidence that
its consumer uses it. No attention quota, hand-set gain floor, hidden oracle,
phase timer or arbitrary extra solver pass substitutes for a sound dataflow.

## M4c1 source and consumer map

| Producer | Consumer | Contract |
|---|---|---|
| Prepared causal image packet | `ObservedEntityHistory` | Exact source offsets/context masks; current content normalization reused, no future argument. |
| Existing inverse flow on later cells | `pull_causal_history` | Compose maps at intermediate coordinates; image-chart units, duplicate step is identity. |
| Earlier-source forward confidence/occlusion | History inverse composition | Read after inverse displacement on its actual source chart; neither is a validity mask. |
| Source-mask interpolation | Temporal relation features | Partial coverage is retained, never tiny-mass normalized or promoted to physical visibility. |
| Camera-local temporal features | Full current-candidate quadrature | Project before expansion; no barycenter or top-k support collapse. |
| Temporal candidate context | Real K competition and slot updates | Alter the actual exported entity law, not an unused metric; independent current target unchanged. |
| Current facts | S/coarse/W/P, compact deployment cache | No new dense source tensor or mutable memory in an ODE/W rerun. |
| Config/ABI | Save/exact reload/deployment loader | Explicit new mode; source file included in import-closure identity. |

M4c1 chooses a bounded observation-derived association path before considering
persistent recurrent identities. Global physical objectness and cross-window
re-identification are NOT declared complete. Do not promote a new field, an
action gradient, or a passing structural test into a learned tracking claim.

## M4d producer/consumer closure

The observed-history packet owns the latest inverse flow and its physical source
gap. `ImageLogMeasure` owns the current per-entity/per-camera spatial law. Their
expectation replaces the stale anchor in `DenseObjectGrounder`, `ObjectFactSet`
and compact `ObjectWorldBelief`; actual S/W consumers see that same exported
value. Ordinary source/config/deployment identities record the new read. Tests
perturb old anchors, verify spatial gradients and action-only gradients, check
real S/W hooks, reset, CPU BF16 and exact checkpoint/deployment reload. The raw
old anchor remains available only to old-mode controls. No source observation
is erased because correspondence confidence, coverage or camera mass is small.

The corrected M4c focused file has 33 cases; its full local inventory finished
with 678 passes and one skip, no failures/errors. The additional M4d acceptance
belongs to its own exact-source report, not this preceding correction result.

## M5a producer-consumer closure

| Producer | Actual consumers | Ownership / lifespan |
|---|---|---|
| S protected goal + timestamped history + G facts | SharedTargetBinder | One FP32 K+null law per physical observation; no oracle or future target. |
| Current attributes, per-camera coordinates/roles and robot state | Internal evidence reads in S/coarse/P1 | Multi-key per-object attention; target mass only after internal reading; no independent target softmax. |
| Shared K mass | P2 semantic K and conditional geometric cameras | Common object marginal for every numerical query; interval/view choice does not replace target identity. |
| P1 operated-target reader | Current factual detail -> real action decoder | Added after clean residual subtraction; optimizer/checkpoint owner registered; no dependency on nonzero W. |
| Current per-view displacement/rate | S state-change evidence | View-aware features before pooling; task-start cumulative progress is explicitly not inferred. |
| Config, source closure, target metadata | Factory/checkpoint/deployment | Distinct new identity, old reader-local path retained unchanged; no strict=False migration. |

The previous M5 recovery ZIP contained zero source files. This unit was rebuilt
from the independently verified M4d archive; earlier unarchived M5 claims are
not used as test evidence. Final artifacts record executed checks, source hash,
unsupported-runtime versus supported CI, and whether publication occurred.


## M5b producer-consumer closure

| Producer | Consumer | State and supervision boundary |
|---|---|---|
| CALVIN declared source/context start | Native state encoder and grouped DINO cache read | One causal instruction origin; independent of random/sliding center and future-support rows. |
| Instruction event and current observation | Online owned reference | New reference on reset/new event, not each action or ODE node; unchanged instruction can be explicitly restarted. |
| Current G entity/query and same camera charts | S `InstructionReferenceRead` | Shared query for current/start; visual difference separate from missing/uncertain match status; no old K IDs. |
| Per-view relation and same-chart robot state change | Shared target marginal -> S intervals | No raw cross-camera vector average, no real-mass renormalization, no reference-age neural feature. |
| Real S intervals | Existing coarse/P1/P2/P3/bottom | Actual policy-only gradients and optimizer ownership required, not merely fields/logs. |
| Reference schema and graph selection | Config, dataset, checkpoint/deployment | Explicit new source identity; undeclared or incompatible reference contracts rejected. |

Half-storage regression exposed an existing G FP16/FP32 boundary mismatch.
The neural observation ingress owns the cast once; old FP32 behavior is checked
across the two actual source trees. Future raw targets are not learned anchors.
Full regression, source hashes and supported-runtime acceptance are distinct.
M4c2 persistent physical association remains open, as do subsequent S/W/P tasks;
this finite observed anchor does not silently create recurrent object states.


## M6a/b/c source preservation and next dependency

M6a pairs real controls with observed future targets without exposing them to
online policy. M6b preserves actual candidate-control support through P2 before
keys and common/residual pooling. M6c supplies the same current robot observation
to candidate/supervised/refinement W and learns per-view relations BEFORE pooling.
These subunits are source checkpoints, not whole-stage behavior acceptance.

| Producer | Real consumers | Isolation/lifetime |
|---|---|---|
| Current authoritative history.state + compact G facts | W initial, supervised, outer refinement | Same observable feature chart, no target/goal/reference/future-state input |
| Named per-view coordinates, positive source-gap and motion | RobotObjectRelationEncoder | Per-view nonlinear features before fusion; missing is not stationary |
| Pooled current relation and per-view relation | Generic W interval base, typed geometry mean AND covariance | One private W1/W2 calculation, no ODE-time neural recomputation |
| Candidate action prefix length | P2 keys, common/residual, spatial/temporal read | Unknown control does not become a zero-motion hypothesis |
| Matched real 48-step controls | Training-only supervised W2 loss | No replacement of online candidate worlds |

Next core decision: the fixed interval 16..32 cannot provide a correctly labelled
16..24 controlled prediction. Under M6b the far W2 fields therefore have no policy
authority for a 24-row proposal (W2 remains trained by actual future controls).
Jointly redesign the query/control time grid and its Teacher/data/S/P consumers,
or supply an explicitly learned longer candidate continuation with matching
supervision; do not add zero controls, relabel timestamps, or remove the source
domain mask. Resolve this in M6 rather than calling far-horizon planning complete.
Persistent physical identity, S desired-operation targets and P3 execution-error
state remain separate open design work. Do not introduce deployment-only state.
