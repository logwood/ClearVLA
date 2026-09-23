# ClearVLA current mainline consolidation and refactor plan

Updated: 2026-09-19 UTC

Status: planning only.  This file orders future repository and source work; it
does not authorize a merge, branch/worktree deletion, checkpoint migration,
training launch or architecture change.  The current working tree remains
under review.

**Scope:** future work and decision gates for the current workspace. Per-run
progress is a dated observation in the handoff. Historical restrictions apply
to their named task/experiment; this plan neither revokes nor extends a later
explicit user instruction. Old branch audits below must be revalidated before
any Git operation.

This is a compact living plan.  Architecture truth is in
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md),
open behavior questions are in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md), and volatile
process state is in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
The former 1,500-line recovery/modularization/B-spine execution narrative is
recoverable at Git commit `b8163cb`; completed proof steps are not repeated
here.

## Planning baselines and dated provenance

| Surface | Fact and evidence scope | Planning consequence |
|---|---|---|
| Git identity | At the 2026-09-04 repository audit, the active checkout and both verified remote refs were `b8163cb`; the working tree has unpublished changes on the historically named `codex/schema29-mainline`. | Treat that branch as the current WIP line until its changes are preserved. Make `master` the sole formal trunk afterward because it is already the remote default; re-fetch and compare immediately before any branch action. |
| Active default | Manifest format 30, component layout 2 and the Schema28-core recovery behavior are the default.  The branch's `schema29` text is historical. | Do not infer behavior, checkpoint compatibility or release status from a branch/run name. |
| Pen evidence | The recovery Pen E8 curve is complete and is the current comparable behavior baseline. | The old “wait for E8” phase is closed.  Remaining far-horizon/event weakness stays in the issue ledger and does not automatically select a mechanism. |
| RDT evidence | The recovery artifact at `0973f192` declared a previous-command gripper boundary but used current qpos in loss/evaluation. | Its event/F1 surface cannot close the RDT contract.  Only a newly initialized run from the corrected boundary can replace it. |
| CALVIN evidence | A frozen E1/11012/Q5 checkpoint now has matched target-position, language and internal-path probes. Target-relative response and color-instruction use are weak in those bounded checks. The W camera and full-action-sequence conditions are implemented opt-in source units, not accepted learned behavior. | Use the six-task colored-push comparison to decide target-routing work; do not promote a source gate, a single pair of failures or a new branch name into an outlet claim. Broader CALVIN closure, including `open_drawer`, remains separate. |
| Optional B-spine | The B-spine implementation is opt-in and its detailed feasibility narrative is historical evidence. | It is a deferred experiment, not the next automatic mainline version and not a reason to retain migration-only golden constraints. |
| Working tree | The root contains several WIP lanes, including CALVIN outlet work, benchmark/simulation work, physical/B-spline assets, dependency/script changes and history cleanup. | Isolate distinct behavior changes, while allowing behavior-neutral cleanup to accompany the owner it clarifies.  Never promote an unclassified dirty tree as one commit. |
| Protected provenance | The `pre-schema29-root-promotion-20260901` stash is based on `codex/v94-latent-ownership-execution`. | Keep the branch and stash untouched until the stash is explicitly inventoried and resolved. |

## Documentation ownership

Repository memory follows one-owner rules just like source code:

| Surface | Owns | Must not own |
|---|---|---|
| Architecture contract | Current graph, typed boundaries, accepted outlet semantics and invariants | Epoch narration, active PIDs, old-version deltas or feasibility reports |
| Current issue ledger | Unresolved question, latest comparable observation and source-entry condition | Closed repair diary, full probe output or execution checklist |
| Repair plan | Future repository/refactor order, safety gates and completion criteria | Current architecture truth or volatile run state |
| Active handoff | Timestamped PIDs, paths, latest process health and immediate next action | Architecture decisions or permanent experiment history |
| Package README | Stable package purpose, public API, ownership boundary and entry points | Release claims, live run state or future design backlog |
| Archive/index | Historical reasoning and retrieval pointers | Authority over the current source |

Maintain these roles in place. On 2026-09-19 the documentation correction
separated W from P2's direct S input, fixed initialization vocabulary, added
source/behavior status distinctions, replaced the stale live handoff with dated
task scopes, and marked the WIP map's old next steps superseded. It changed no
model, training, checkpoint or experiment selection.

Each future update must name its scope: current source/config, an identified
run/checkpoint, a proposed unit or a historical snapshot. A header date does
not refresh every measurement in a document. Prefer links to the owning record
over a second live status table. Continue the cleanup in place:

- keep the R1/R2 and conversation ledgers explicitly historical and outside the
  current authority chain;
- replace the temporary B-spine early audit after its next decision checkpoint
  instead of growing a second architecture contract;
- treat the ignored v120_bspline_reintegration directory as a local historical
  prototype; never apply its conceptual patch to the current mainline or cite
  its old current-status language;
- prefer Git history over a new versioned archive document when removing
  obsolete narration;
- update a package README only for a stable interface or command, and put
  learned results in the issue ledger or raw run record.

## Repository convergence

Repository convergence precedes further architectural refactoring.  It is a
review sequence, not permission to execute destructive Git operations.

### Preserve and classify the root WIP

Classify the root changes into review lanes:

1. CALVIN sampler, outlet-action and world-conditioning semantics;
2. benchmark and simulation packages plus their tests;
3. physical chart, B-spline and robot/URDF assets;
4. dependencies, launchers and operational scripts;
5. architecture-memory and historical-document cleanup.

Generated `.tmp`, audit outputs, caches and copied experiment artifacts do
not enter these commits.  Each changed area must state whether it changes
active behavior, an optional component, operational tooling or documentation.

These lanes do not require one commit per list item.  A behavior-neutral rename,
owner move, utility consolidation, documentation correction, legacy archival or
launcher cleanup may accompany the nearest semantic/owner unit when all of the
following hold:

- it concerns the same owner, interface or lifecycle;
- it does not change tensor math, defaults, data selection, loss composition,
  checkpoint/deployment ABI or runtime call count;
- its compatibility handling and validation are visible in the same review;
- it can be reverted with that unit without pulling out an unrelated feature.

Large independent additions such as benchmark/simulation packages should still
stand alone when bundling them would obscure review or ownership.  Two distinct
behavior hypotheses must never be hidden inside one “cleanup” change.

### Promote one formal trunk

After the WIP units are preserved and reviewed:

1. advance local `master` only to the accepted ordered commit sequence;
2. switch the root worktree to `master`;
3. verify that the canonical manifest/config/launcher resolve from that tree;
4. retain `codex/schema29-mainline` until the new `master` tip and all
   unpublished WIP are independently recoverable;
5. remove the duplicate branch only after the ancestry and clean-worktree
   gates pass.

Any fetch, push or remote-branch deletion requires a new remote comparison;
the local remote-tracking snapshot is not proof of current server state.

### Branch and worktree disposition

Revalidate every row immediately before acting.

| Ref | Current disposition |
|---|---|
| `codex/schema29-mainline` | Current capability/WIP line.  Promote accepted commits to `master`, then delete only when redundant. |
| local `master` | Behind the local `origin/master` snapshot.  Fast-forward after WIP protection; do not merge the old local tip as a side line. |
| `codex/rdt-multitask-prep` | Do not merge wholesale.  Preserve only explicitly selected documentation/provenance, then archive and remove the branch/worktree. |
| `codex/schema28-core-recovery-pen-20260903` | Retain an immutable provenance tag for the formal recovery run, then remove the redundant branch/worktree. |
| `codex/schema25-r1-replay` | Code is already patch-equivalent in mainline; first preserve or reject its dirty/staged documentation, then remove it. |
| `codex/schema28-estimator-gate` | Estimator code/tests are already patch-equivalent in mainline; preserve decision-relevant documentation only, then archive/remove. |
| `codex/v94-latent-ownership-execution` | Keep while the protected stash depends on it. |
| `codex/v86-slot-controller` | Extract the small real source/test/launcher WIP from line-ending noise before removal. |
| `codex/v76-recovered` | Historical ancestor.  Verify its retained tag/provenance, then remove the branch. |
| `origin/codex/rdt-data-adaptation` | Remote ancestor and later deletion candidate; revalidate against the server before deletion. |

Before removing any worktree or branch, require: clean/understood status,
recoverable unique commits, preserved intentional untracked files, no dependent
stash, a recorded replacement ref where needed and an ancestry/patch-equivalence
check against the chosen `master`.

## Active work order

The current cross-task sequence is owned by the
[decision order](CURRENT_MAINLINE_ISSUES.md#decision-order): close the two
independent CALVIN result decisions, prepare the exact-order reader in parallel
and integrate it at the agreed milestone, then select any further structural
unit from the remaining behavior question. The sections below are stable work
packages and entry conditions, not a second priority ranking or launch queue.
Outlet admission gates apply only to the selected artifact/claim. Broader
identity/naming/migration refactors are not prerequisites for the bounded
reader unit. The former first Pen trace package is now under deferred work.

### 1. Close outlet contracts before changing the shared core

- RDT: treat the old recovery event metrics as incompatible. A corrected,
  identified source and new initialization are required to replace that
  behavioral reference if RDT is selected; this is an outlet admission
  condition, not the next training task in the current scope.
- CALVIN: keep the shared value/adjacent-difference 18-D core. The current
  six-task colored-push comparison is the matched target-binding test: require
  complete comparable curves and reset-autonomous closed-loop results before
  promoting a learned source change. The outlet-scoped centered-command
  sampler, W projection and binary gripper isolation retain their own gates;
  `open_drawer` remains a broader outlet-closure check, not a substitute for
  target-relative pushing.
- LIBERO: keep the shared seven-dimensional codec untouched while closing the
  benchmark edges.  The converter must keep filename-derived evaluator
  `Task.language` separate from any BDDL `:language` alias, reject malformed or
  misaligned native arrays before publication, require the official BDDL task
  inventory as the proof of evaluator-language identity, and
  stage/audit/atomically publish the complete root.  The evaluator must
  preserve fixed-init state, discard success/done during the official
  five-step physics warmup, execute
  exactly the first clipped row of each 24-row chunk, and feed that executed
  row back as the next action-state.  Unit/fake-environment gates cover these
  boundaries, fixed init states remain first-N rather than cyclic, and resume
  identity includes both affine normalizer digests plus a complete per-rollout
  action audit.  Evaluate the 90/10 suites separately instead of treating the
  `libero_100` archive union as an evaluator suite, and never label a smoke
  bridge as official.  Image size and the used BDDL/fixed-state content belong
  to resume identity; a relocated bridge endpoint does not.  Intermediate
  atomic results remain explicitly incomplete until final bridge identity is
  revalidated.  A real closed-loop score remains an external validation step,
  not a network-core change.
- Pen: retain the completed recovery curve as the shared-core guard.  Do not
  change S/W/P/outer closure merely because far/event behavior remains weak;
  first select the producer-to-consumer failure with matched evidence from the
  current issue ledger.

### 1a. Resolve CALVIN target routing across S, W and P2

The latest bounded evidence is in `CURRENT_MAINLINE_ISSUES.md` and
`artifacts/shared_s_candidate_20260918/`. It separates three verified
representation limits from their still-open behavioral attribution:

- The legacy four-mean W condition aliases action order and the first three
  rows. The opt-in 24-row condition closes that input observability limit
  structurally, but its learned correction still needs matched validation.
- The legacy W camera producer has limited view-specific input. The opt-in B1
  condition addresses that producer boundary; it does not alter P2's reader or
  establish world-frame transport. Two initial snapshots do not show strong
  cross-camera vector cancellation, so a P2 fusion rewrite is not automatic.
- Direct typed-S reaches P2's interval terminal only after K/K*C spatial
  pooling. A real-checkpoint typed-only transplant leaves that spatial read
  unchanged and barely changes one-node velocity, but the S donor itself
  changes little under the color swap. Real geometry-read/effect transplants
  do change full-Q5 actions. Neither result identifies P2 alone, absolves the
  bottom, or predicts that moving S before pooling will improve contact.

The final evidence-completion round is closed. The following is a decision
order for later selected work, not a request to reopen that round or start a
four-cell combinatorial ablation. Reuse its evidence and retain its limits;
new diagnostics or runs require their own concrete task scope.

1. Compare each already implemented opt-in W condition against the same
   identified E1 initialization and legacy control, with matched six-task
   split, instruction bank, normalizers, Q5 schedule and evaluation layouts.
   Keep camera and action-sequence claims separate. If a combination is later
   selected, name it as a new interaction hypothesis rather than retroactively
   attributing its score to either individual change.
   Keep early checkpoint screening separate from final model selection. A
   saved historical V1 checkpoint can receive a bounded behavioral screen
   while B1/sequence training continues; preserve its original model/deploy
   contract rather than relabeling it as the Q5 control. The read-only candidate
   review is `artifacts/closed_loop_selection_20260919/REVIEW.md`. No evaluation
   is launched by this plan, and existing E1 results should be reused where
   identity and protocol match. Comparing an additionally trained structural
   arm to the original E1 is useful for selection but is not single-factor
   causal attribution without an equal-training-budget legacy control.
2. Before a target-K oracle or S/P2 source change, verify entity-to-K mapping
   from joint chart assignment in each scene; slot permutation alone is not a
   grounding error. Under fixed observation/history/noise, trace target-word
   change through S's K-preserving evidence, P1/coarse, W, P2 spatial and
   interval reads, then physical action. If the mapping gate fails, retain only
   representation/path conclusions, not target-identity success claims.
3. The selected source unit is now implemented as
   `top.p2_spatial_intent_mode=shared_target_prior_v1`. One shared FP32
   `[B,I,K]` address is derived from S's typed semantic/appearance/geometry
   evidence by an exact-zero, RNG-neutral owner. The same K address conditions
   semantic K and geometry K*C before pooling; W alone still supplies values
   and G/W retain support authority. Post-pooling typed S is retained only for
   interval selection, so there is no duplicate target-K choice.
   The unit preserves K-to-camera correspondence, both spatial types, four
   intervals, no-null physical terminal and exact neutral initialization. It
   rejects supported NaN/Inf, masks unsupported K, and adds no raw color label,
   bottom-language bypass, hand-set gain, entropy quota or contact-triggered
   stage rule. Its explicit old-checkpoint initialization adds one `[1,3]`
   zero weight and starts a fresh optimizer/schedule/RNG; behavior promotion is
   still pending.
4. Treat camera-aware P2 value fusion as a separate reader decision. First
   establish whether the chosen W producer supplies useful view-specific
   transport that the reader loses. Merely moving the same shared linear map
   before averaging is algebraically neutral. Any new view-conditioned map
   must bind declared camera roles, preserve zero physical input and invalid
   camera isolation, and must not claim unprovided calibration.
5. Close every chosen source unit with forward and owner-VJP tests, K and
   joint camera/role permutations, support/all-invalid and zero-effect cases,
   checkpoint/migration round trip, unchanged training and Q5 call counts,
   and Pen/RDT non-regression. Behavioral acceptance additionally requires
   matched full-Q5 action comparisons and reset-autonomous six-task rollouts:
   target-relative approach at the relevant physical stage, actual push
   direction, non-target contact and success. Training uses only the train
   split; moving a target image without regenerating its expert action is not
   a valid augmentation.

### 1b. Integrate the exact-order visual reader independently

The DataLoader research and coordination with 新主执行者 are recorded in
`artifacts/dataloader_impl_20260919/INTEGRATION_PLAN.md` and `RESULTS.md`.
The first source unit is an explicit opt-in **pread reader for the existing
NPY cache layout**, with legacy mmap as the default. It retains existing cache
admission, Dataset math, normalization, sampling, batch/frame/camera order and
DataLoader RNG ownership. Phase-layout conversion, compression and
cross-annotation deduplication remain separate decisions.

The isolated preparation base was
`aa798d12ee7abe0261bc252ba5c30d4d95b6bf4a`, with isolated branch
`codex/calvin-loader-pread-20260919` at commit
`c77420adb9b52e56d089b95b9df171df4d430223`. Its bounded reader was ported into
the current working source on 2026-09-20 without enabling it by default.
`NpyRowReader`, decoded-image and token-store routing, config serialization,
ordered/repeated-row parity, camera order, pickle/handle lifecycle and focused
tests are present. The explicit combined initialization
`p2_shared_target_prior_pread_v1` admits only the shared-target source unit plus
`data/loading.py`, `data/token_store.py`, `vision/decoded_image_store.py` and
`vision/npy_rows.py`; the standalone P2 migration does not admit reader drift.
This source integration is not production deployment. Check the complete real
frozen source identity before upload, and do not widen the allow-list to absorb
unrelated sampler, dataset, engine, dynamics or simulation drift.

Production activation still requires selection of the actual frozen checkpoint
and source, followed by full admission below. Enable the reader only in a new,
explicitly identified component-initialization run. An in-progress experiment
does not change reader midway, and the new run is not exact continuation of the
old one.

Admission requires the real E1 normalizers and complete episode/index/cache
identity, representative full-batch byte equality at worker=0 and production
worker/prefetch settings, and the original seeded sample/RNG stream. Distinguish
matched new-run initialization and a deterministic optimizer-step check from
actual recovery, which additionally requires sampler cursor, worker RNG,
next-batch and complete state-continuation equivalence. Measure cold and warm
end-to-end throughput, memory and descriptors separately. The completed
six-episode I/O experiments do not certify full-data recovery or training
speedup. Identity/content mismatch is a hard failure; legacy fallback is an
explicit recorded configuration, never a silent mid-run switch.

### 1b-HDF5. Separate episode admission from sample reads

The current `pread` unit does not remove the remaining cold-start bottleneck:
`load_episodes` still opens every manifest-selected HDF5 episode and
materializes action/state arrays before the first DataLoader batch, while the
CALVIN raw overlay also validates its source trajectory/frame chart during
admission. The active GPU4 B8 smoke is intentionally being left untouched
while these phases are separated by timing; its long external-disk wait is a
loader-admission observation, not a model failure. This is a separate loader
issue and does not authorize a new training run.

Prepare an opt-in `hdf5_indexed_lazy_v1` reader with these boundaries:

1. Build or load a compact, immutable episode index containing root-relative
   identity, resolved dataset keys, shapes, lengths, contract attributes,
   normalizer/statistics digests and source-file size/mtime/hash. Do not read
   full action/state arrays during index admission.
2. Keep the existing manifest order and `torch.Generator` ownership. Workers
   open HDF5 files only when a sample is requested, using a bounded per-worker
   handle/row cache. Do not let lazy reads change episode, frame, camera or
   repeated-row order.
3. Keep raw-overlay endpoint validation and terminal-padding semantics
   fail-closed. A stale or incomplete index, invalid dataset key, changed file
   identity or non-finite selected row must reject the lazy path rather than
   silently fall back midway through a run.
  4. Add a loader-only matched verification: eager versus lazy first batches
     must have identical episode identities, normalized bytes, camera/token bytes,
     masks, normalizer digest and sampler/RNG continuation. Record cold index
     time, first-batch time, warm steady-state throughput, worker RSS and open
     descriptors. No model forward or optimizer step is needed for this gate.

  The implementation boundary is deliberately dataset-agnostic: an immutable
  source fingerprint plus an ordered `RowRequest` and `EpisodeRowReader`
  protocol is the shared contract in `clearvla/data/index_contract.py`. HDF5
  supplies the first adapter; NPY, sharded archives and future remote stores
  can implement the same identity, fingerprint, field-shape and ordered-row
  operations without changing the sampler or model-facing batch contract. A
  backend may canonicalize rows for its storage library, but it must restore
  the caller's order and reject stale source identity before returning data.

  The legacy eager path remains the default until this gate passes. The index
  format and lazy reader must be reviewed as one source unit; do not claim a
  training speedup from the current NPY `pread` measurements alone.

  The latest read-only probe on 32 real CALVIN train episodes (fixed seed 17)
  preserved identity order and selected row bytes. It measured 32.91 s for
  the one-time metadata index, 6.2 ms to reload it, 144.4 ms for eager
  admission, and 56.7 ms for the matched selected-row lazy read. Three warm
  repeats with a bounded worker-local handle cache were 89.3 ms versus 96.3 ms
  for the open-per-call control. These numbers support warm reuse but do not
  establish end-to-end training speed; raw-overlay admission, normalizer
  fitting, profile projection and production worker/RNG continuation remain
  separate gates.

  A second loader-only gate on the same 32 episodes also matched the eager
  action/state/action-state bytes and both serialized z-score normalizer
  digests exactly (`bcf08bdf97bf40e1eb429436ed800a6adb409f40751d24d13b12910a0467923d`
  and `d0e9beee83e162615a99b3917a91bad89ec40165518eb689776dbede42fda031`).
  The lazy row phase took 87.6 ms after admission. This closes the prototype's
  normalizer/field-byte gate for the sample, but not the full E1 terminal,
  overlay, or worker-resume gate.

### 1c. Evaluate the historical V1 before deciding whether to retire its trainer

The authorized bounded panel is complete. It evaluated the frozen
object-binding V1 E4/105308 checkpoint (`34ee78e98fd8cb891f30a4d23fe33ecfebd50271`,
checkpoint SHA-256
`327e4a3d1a946568330b76c6b3fc5f57e67c2aa824d6c1c927759d47734d7119`) on 34
independently reset trials across 14 tasks under its original direct-command,
uniform five-step deployment contract. Official CALVIN oracle success was
6/34; push was 2/18 with 11/18 target contacts (left 9/9, right 2/9), and
lift/drawer were both 0/4. The result is retained as a frozen target-routing
and collateral-interaction baseline, not as a promotion candidate or a matched
language counterfactual. The local reconstructed artifact and evidence limits
are in `artifacts/calvin_e4_20260919_download/analysis/summary.md` and
`statistics.json`.

This remains separate from the B1/sequence model-selection gate. The panel
does not authorize stopping existing training, expanding the panel or
activating the new reader; any such action remains an explicit user decision.

### 2. Separate data and run identity

Split the current composite identity into explicitly named records:

- raw episode inventory;
- split membership;
- outlet/profile metadata;
- state/action normalizers;
- decoded and DINO cache identities;
- language artifact;
- model/component resume ABI;
- optimizer, scheduler, sampler cursor and named generator continuation state;
- source commit and complete source snapshot as provenance.

The current checkpoint `data_state` must either become a validated/restored
`SamplerContinuationState` or be decomposed into provenance fields.  A value
that is merely saved but neither checked nor restored must not be called resume
state.  Likewise, a digest containing profile metadata must not be described
as a raw-data checksum.

### 3. Separate semantic names and owners

The target rule is one name per semantic axis and one owner per lifecycle.

| Overloaded surface | Problem | Target |
|---|---|---|
| `bottom.arm_flow_mode` | Names CALVIN source semantics, evaluation motion semantics and the extracted V120 internal field chart, although the core is forced back to `legacy_independent`. | Split core representation, outlet action semantics and W projection.  Evaluation derives its definition from the outlet contract. |
| `BottomConfig` | Mixes core action field, terminal gripper behavior, outlet semantics and optional B-spine selection. | Compose separate core-action, outlet-action and execution-bottom configuration records. |
| `OutletAdapter` | Owns codec proxying, normalizer sidecar, native conversion, W projection, dynamic sanitization, terminal finalization and metrics. | Give these to the canonical codec, outlet chart, `WorldConditionAdapter`, outlet finalizer and outlet-aware metrics respectively.  A facade may coordinate but owns no formula. |
| Schema labels | Mix behavior, manifest format, layout, ABI, experiment lineage and branch names. | Serialize `behavior_contract`, `manifest_format`, `component_layout`, `component_abi`, `outlet_profile` and `source_revision` independently. |
| `CheckpointIdentity` / `source.digest` | Mix provenance with resume compatibility; canonical source-text changes are a hard veto even when the semantic contract is unchanged. | Separate `ProvenanceRecord`, `ResumeContract`, `DeploymentABI` and `DataIdentity`.  Source text is audit evidence, not the sole compatibility decision. |
| `fresh run` | Can mean new weights, new lineage or simply an empty output directory. | Use explicit lifecycle modes: `new_training`, `exact_resume`, `validation_replay` and `component_initialization`.  Keep overwrite protection in a separate run-directory guard. |
| golden/exact-byte/order gate | Mixes a one-time layout migration proof with permanent architecture semantics. | Freeze the old cross-source capture as replay-only evidence; do not make its historical constructor order or byte-for-byte capture a permanent architecture definition. |
| policy construction | `ClearVLAMainlinePolicy` constructs temporary monoliths, detaches/re-registers children and keeps legacy ordering while also acting as the live composition root. | Construct final owners once through the component factory.  Isolate legacy key/order translation in checkpoint compatibility code. |
| package/launcher names | `v120_core`, duplicated `clearvla/policy`, many `current_v*` scripts and a root `run_current_policy.sh` that launches V48 all appear authoritative. | Keep one role-named active package and canonical mainline train/smoke entry points; place historical implementations and launchers under an explicit legacy/archive namespace. |

The intended CALVIN terms are:

    core_arm_representation = value_adjacent_difference
    outlet_action_semantics = calvin_relative_tcp
    world_condition_adapter = calvin_relative_command

Pen and RDT select their own outlet semantics and W adapter without changing
the shared core representation.  A compatibility reader may map an old
`arm_flow_mode` only when the old artifact makes all three facts unambiguous;
new artifacts write only the separated fields.

### 4. Remove migration mechanics from live construction

Layout 2 is already the current registered hierarchy, so the old atomic
modularization project is not an open 800-line task.  The remaining cleanup is
smaller:

1. freeze the layout-1 -> layout-2 key map and equivalence report as history;
2. construct final registered owners directly in the accepted initialization
   contract;
3. move legacy key translation and any required old-order handling outside the
   policy forward/construction path;
4. verify state coverage, owner uniqueness, forward/backward behavior,
   optimizer state and deployment lifecycle;
5. remove detach/re-register compatibility machinery after retained
   checkpoints have a tested reader or explicit archive-only status.

This is a behavior-preserving structural unit.  It must not carry a CALVIN
semantic change, B-spine enablement, loss adjustment or numerical gain.

### 5. Historicalize migration-only controls

After the structural unit above closes:

- move the V76-V88 golden harness, scripts and immutable reports to explicit
  historical replay scope;
- remove the old cross-source constructor-order and byte-for-byte capture from
  the standing architectural definition;
- keep the empty-output-directory check as filesystem safety;
- retain source snapshot and commit for provenance;
- decide resume from explicit semantic/config/component/data/optimizer/
  scheduler/sampler contracts, with an explicit reviewed compatibility
  declaration whenever source provenance changes.

Neither “source digest changed” nor “the output directory is non-empty” is, by
itself, a statement about model architecture.  This separation does not weaken
fail-closed resume validation or permit output overwrite.

## Deferred work

### Pen structure questions — deferred

This is a retained work package, not a predecessor of the current CALVIN
comparison or reader preparation. Reopen it only when a Pen/shared-core
question is selected. Keep the following evidence classes separate:

- **Closed:** `ffn_expansion` propagation and producer-owned S/coarse/W validity
  passed their named source gates. The typed S common/residual reverse path is
  explained by its accepted lossless decomposition; it is not an omitted owner
  connection requiring another repair.
- **Unresolved representation/interface questions:** dynamic P1 basis detail at
  its two consumers, six-channel gripper fields versus deployed decode,
  current-G3 transport reused across future intervals, and the public
  future-length/support contract. These have different owners and are not one
  accepted structural defect.
- **Insufficient attribution:** finite gradient spikes, weak geometry address
  utility and a near-uniform typed-P2 router do not select a mechanism. The
  rejected geometry/gripper combination cannot isolate either factor's effect.

If one question is selected, first use its retained evidence to state the
specific forward/reverse/action hypothesis and missing acceptance condition.
Use a bounded matched check only if that condition needs new evidence; do not
schedule all earlier traces again. Keep a chosen repair in one semantic unit
and retain the existing Pen guard before considering combinations. No fixed
gain, route/entropy quota, loss-weight change or extra ODE/W pass follows from
these observations alone.

### B-spine

B-spine remains an optional `execution_bottom` experiment.  The already
selected Pen run may supply evidence for that bounded candidate, but it does
not name a new default or authorize another sweep. Do not start a follow-up
variant until the repository has one clean trunk, the active outlet contracts
are closed and the user explicitly selects it. If continued, use only this
compact ladder:

1. one read-only cross-outlet chart/locality probe;
2. one opt-in bottom implementation with the original full-resolution path
   retained;
3. focused forward/reverse, checkpoint, real CUDA and lifecycle gates;
4. one matched Pen pair, rejecting gains caused by event/detail suppression.

The arm-private reader control is now available as the second bounded
implementation for step 2:
`configs/mainline/object_intent_dynamics_323_pen_bspine_arm_private_reader.json`.
It shares the fixed arm coarse chart with the coarse-context candidate but
routes that view through an arm-only terminal correction, so the comparison can
separate temporal arm context from shared arm/gripper fusion. The implementation
does not change the codec, objective, ODE schedule, W rebuild or output ABI;
only a matched run and the listed attribution/gradient gates can determine its
disposition.

No learned knots, output-space spline writer, extra ODE/W loop, hand-tuned
gain, quota or smoothing objective belongs to that first experiment.

### Benchmark and simulation

The benchmark/simulation work is tooling until it has independent tests,
explicit external-dependency boundaries and a canonical launcher.  The LIBERO
edge adapter now has focused conversion/evaluator tests and an explicit
deployment identity, but it must not be used to claim learned benchmark
behavior until a real official closed-loop run is recorded.  Keep this outlet
work separate from the shared network-core units; remove direct legacy-lab
dependencies or label them as optional compatibility before promotion.

### Native RDT expansion

Native three-camera, depth and full 14-D bimanual support are later adapter/ABI
units.  The current right-arm two-camera RDT-8 adapter does not establish those
capabilities.

## Validation policy

Every source unit declares its changed semantic boundary before implementation.
Use the smallest applicable gate:

- documentation/config naming: schema/parser and documentation consistency;
- behavior-preserving owner movement: state/key coverage, owner uniqueness,
  focused forward/backward, optimizer and deployment lifecycle;
- outlet semantics: Pen/RDT non-regression plus outlet-specific CPU, real
  CUDA/BF16 owner-VJP, checkpoint and smoke tests;
- learned behavior: complete comparable curve and the matched intervention
  that selected the change;
- branch/worktree removal: clean status, unique-commit/patch-equivalence,
  stash dependency and replacement-ref checks.

Hard stops remain non-finite values, identity/lineage failure, an open loss
ledger, vanished formal parameter-owner VJP, checkpoint ABI violation or
process memory above 22 GiB.  Finite threshold crossings, small geometry
amplitude, early event F1 and capacity warm-up remain telemetry unless tied to
a reproducible behavior or health failure.

Do not combine a connection/ownership repair with a gain, quota, hard event
gate, entropy target, clipping change, loss-weight change or extra inference
pass.  One source unit should answer one pre-stated question.

## Completion

This plan is complete when:

1. `master` is the only formal trunk and the root worktree uses it;
2. every retained branch/worktree/stash has a documented purpose, and redundant
   branches are archived or removed only after their safety gates pass;
3. root WIP is organized into reviewed semantic units, with compatible
   behavior-neutral cleanup allowed to travel with its owner, and generated
   artifacts are excluded;
4. RDT and CALVIN claims match their actual validated outlet contracts;
5. identity fields no longer mix raw data, metadata, source provenance, resume
   ABI and deployment ABI;
6. `arm_flow_mode`, temporary monolith construction and misleading
   `current` entry points no longer exist on the active write path;
7. golden/fresh-run/source-text controls have their intended historical,
   provenance, continuation or filesystem-safety roles rather than
   architectural authority;
8. optional B-spine, benchmark and simulation work remains clearly separated
   from the accepted default until its own gates close;
9. the CALVIN target-routing decision distinguishes W action order, W camera
   production, S/P2 object addressing and camera-aware P2 reading, with no
   opt-in silently promoted from structural checks or one-node sensitivity.

At each closure, update the architecture contract only when accepted behavior
actually changes, update the issue ledger in place, and rely on Git history
rather than rebuilding a chronological diary in this file.
