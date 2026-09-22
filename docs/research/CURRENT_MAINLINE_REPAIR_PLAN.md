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

## Status and dependencies

| Unit | Status | Completion boundary / next obligation |
|---|---|---|
| M0 source baseline | Implemented | Full base archive verified against Git blobs; final tree diff of the two experiment heads recorded; isolated branch and read-only source audit exist. Baseline test and diagnostic scope is in the handoff. |
| M1a real history time | Implemented candidate; validation recorded per commit | Shared source clock; separate state/control encoding; padding/dropout masks through S, coarse and proposal; bounded online history; new config/components/ABI. This is not all of M1. |
| M1b real-tail labels | Implemented candidate; validation recorded per commit | All labelled real current centers and independent action/state/visual support through loader, Teacher, auxiliary/formal losses and validation. Source files/caches and old defaults unchanged. |
| M1c native/feature state | Implemented candidate; validation recorded per commit | One CALVIN rotation-column encoder for current/history/future/online; native statistics and seven-dimensional commands remain separate; explicit new ABI and checkpoint checks. |
| M1d visual source time | Implemented candidate; verification recorded per commit | Source gaps, repeated frames, semantic/raw flow loss support, temporal/global G reads and future queries, S/W rates and Teacher duration use one observable clock. No claim about unprovided external controllers. |
| M2 observation/G1 | Static source review started; redesign not implemented | Raw/DINO source clocks are repaired. G1 already retains the full coarse posterior, but G2 localization consumes moment-based centers; next change must preserve modes in the real candidate materializer, not add an unused copy. |
| M3 G2 | Not started | Entity-consistent attribute/localization refinement and bounded-error recovery/reassociation. |
| M4 G3 + current Teacher identity | Not started | One entity state with per-camera support and causal association; update reconstruction and Teacher references together. |
| M5 S/coarse | Interface adapted only | The history submodule is new; target/operation/progress decomposition, shared entity binding and recognizer supervision still require deep review. |
| M6 W1/W2 | Not started | Match supervised action to observed outcome; separate candidate prediction from demonstration-target training; explicit robot-object relations and known control horizon. |
| M7 P1/P2 | History context adapted only | Shared operated entity, current target facts, view-aware geometry and corresponding candidate consequences. |
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
M2 must consider how G1's full posterior is consumed by G2's actual resampler;
retaining logits in a sidecar alone does not preserve mode information in the
positions used for candidate values. Any next candidate-mode change must also
review fine-count assumptions, P1/P2 reads, property support and Teacher.

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
