# ClearVLA current mainline consolidation and refactor plan

Updated: 2026-09-10

Status: planning only.  This file orders future repository and source work; it
does not authorize a merge, branch/worktree deletion, checkpoint migration,
training launch or architecture change.  The current working tree remains
under review.

This is a compact living plan.  Architecture truth is in
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md),
open behavior questions are in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md), and volatile
process state is in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
The former 1,500-line recovery/modularization/B-spine execution narrative is
recoverable at Git commit `b8163cb`; completed proof steps are not repeated
here.

## Current baseline

| Surface | Current fact | Planning consequence |
|---|---|---|
| Git identity | The root worktree is at `e62b1b7` on the historically named `codex/schema29-mainline`; `origin/master` is its ancestor by four commits at the 2026-09-10 local snapshot.  The isolated `codex/mainline-integration-20260910` branch starts at the same commit and owns convergence work only. | Treat neither commit date nor an experiment branch name as release status. Preserve and review semantic units on the integration line, then make `master` the sole formal trunk; re-fetch immediately before any remote action. |
| Active default | Manifest format 30, component layout 2 and the Schema28-core recovery behavior are the default.  The branch's `schema29` text is historical. | Do not infer behavior, checkpoint compatibility or release status from a branch/run name. |
| Pen evidence | The recovery Pen E8 curve is complete and is the current comparable behavior baseline. | The old “wait for E8” phase is closed.  Remaining far-horizon/event weakness stays in the issue ledger and does not automatically select a mechanism. |
| RDT evidence | The recovery artifact at `0973f192` declared a previous-command gripper boundary but used current qpos in loss/evaluation. | Its event/F1 surface cannot close the RDT contract.  Only a newly initialized run from the corrected boundary can replace it. |
| CALVIN evidence | The current outlet-scoped sampler/W adapter has local and isolated smoke/VJP evidence but no accepted complete formal curve or closed-loop result. | Keep it outside released capability claims until formal behavior and `open_drawer` closure exist. |
| Optional B-spine | The B-spine implementation is opt-in and its detailed feasibility narrative is historical evidence. | It is a deferred experiment, not the next automatic mainline version and not a reason to retain migration-only golden constraints. |
| Working tree | The root contains several WIP lanes, including CALVIN outlet work, benchmark/simulation work, physical/B-spline assets, dependency/script changes and history cleanup. | Isolate distinct behavior changes, while allowing behavior-neutral cleanup to accompany the owner it clarifies.  Never promote an unclassified dirty tree as one commit. |
| Protected provenance | The `pre-schema29-root-promotion-20260901` stash is based on `codex/v94-latent-ownership-execution`. | Keep the branch and stash untouched until the stash is explicitly inventoried and resolved. |

## Documentation ownership

Repository memory follows the same one-owner rule as source code:

| Surface | Owns | Must not own |
|---|---|---|
| Architecture contract | Current graph, typed boundaries, accepted outlet semantics and invariants | Epoch narration, active PIDs, old-version deltas or feasibility reports |
| Current issue ledger | Unresolved questions, latest comparable observations and source-entry conditions | Closed repair diary, full probe output or execution checklist |
| Repair plan | Future repository/refactor order, safety gates and completion criteria | Current architecture truth or volatile run state |
| Active handoff | Timestamped PIDs, paths, process health and immediate next actions | Architecture decisions or permanent experiment history |
| Package README | Stable package purpose, public API, ownership boundary and entry points | Release claims, live run state or future design backlog |
| Archive/index | Historical reasoning and retrieval pointers | Authority over current source |

Prefer Git history over new versioned archive documents when removing obsolete
narration.  Update a package README only for a stable interface or command;
learned results belong in the issue ledger or raw run record.  Temporary
B-spine audits and old conversation ledgers remain evidence, not competing
architecture contracts.

## Repository convergence

Repository convergence precedes further architectural refactoring.  It is a
review sequence, not permission to execute destructive Git operations.

Generate the current factual inventory from the integration checkout with:

```text
python scripts/audit_repository_convergence.py --base HEAD
```

The command is read-only and reports worktree dirtiness, local and
remote-tracking divergence, merge bases, patch-equivalent versus unique
commits, and stashes.  Its output is a transient review input, not a committed
status ledger or an automatic disposition.  A new inventory is required
immediately before any merge, archive or removal.

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

The first candidate prefix is the four-commit operational sequence
`origin/master..e62b1b7`: it changes only README/handoff text,
the standard-library workspace tool and its tests.  Local compilation, Ruff
and the Windows workspace tests pass.  The exact committed `e62b1b7` workspace
tool and test were also exported from the Git object database into an isolated
Linux temporary directory; all 17 workspace tests passed, including the
POSIX-only open-file rename, process-identity and virtualenv-symlink cases,
without modifying a tracked checkout.  This closes the platform lifecycle
gate.  The prefix remains a candidate rather than a release claim until the
shared WIP is protected/reviewed and the volatile handoff is refreshed.

The initial Windows CPU baseline after removing one stale test/API mismatch is
`792 passed, 10 skipped, 4 failed`.  The removed test imported
`_dwell_value_targets`, which existed only at historical commit `82700f3` and
was removed from the runtime at `b2c1fa8`; restoring unused runtime code would
have hidden the test drift.  The four remaining failures are all in the
historical hierarchical-controller experiment: family-attention diversity,
scope validation, neutral update-logit initialization and exact legacy-forward
identity.  They are an explicit baseline debt, not an allowed-failure list and
not evidence against the current default mainline.  Resolve that experiment as
a separate semantic unit before claiming a fully green repository baseline;
do not import the active `v86-slot-controller` WIP wholesale.

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
| `codex/mainline-integration-20260910` | Sole convergence staging line.  Admit reviewed semantic units only; do not run model experiments from it or call it the formal trunk. |
| `codex/schema29-mainline` | Shared root capability/WIP line.  Preserve its unpublished units independently; do not write integration-only changes back into its dirty worktree. |
| local `master` | Behind the local `origin/master` snapshot.  Fast-forward after WIP protection; do not merge the old local tip as a side line. |
| `codex/training-acceleration` | Active, divergent acceleration line with local WIP.  Admit only completed equivalence-gated units after its owner closes the worktree; never merge the dirty branch wholesale. |
| `codex/pen-rdt-shared-core-schema28-20260908` | Formal shared-core experiment provenance plus separate local WIP.  Preserve the run ref; extract any accepted outlet/loss change as its own reviewed unit. |
| `codex/pen-gripper-anchored-ablation-20260909` | Single-variable Pen persistence experiment.  Retain as provenance until the run is disposed; it does not select the default persistence contract. |
| `codex/pen-rdt-shared-core-20260908` and `codex/remove-bspine-keep-solver-20260906` | Both currently point at `c7eccde`, but one worktree has unpublished changes.  Treat them as distinct retained state until WIP and run provenance are resolved, then keep one replacement ref if still needed. |
| `codex/pen-geometry-gripper-repair-20260907` | Coupled geometry/gripper candidate.  Do not merge wholesale or infer either intervention independently from the combined result. |
| `codex/pen-bspline-routing-20260906` | Optional B-spine experiment line.  Keep outside the default until its matched evidence and interface gates close. |
| `codex/rdt-multitask-prep` | Its four unique commits are reviewed and none is an admissible standalone early-convergence unit.  `f6d2a0d` couples accurate historical audit relabeling to an unapproved reversal of the current threshold/config/launcher policy; `2bebd43` is a one-time v1-to-v3 reconciliation producer; and `ffe5c64`/`f8d0c2f` form an untested task/machine-specific gate with `/data`, an eight-GPU inventory, an old workspace command and an old source-scope baseline embedded in its report.  Its own saved result is `BLOCKED`/`DO_NOT_START`.  Preserve the untracked audits, reconciliation, repeated acceptance reports, launch-gate report and `new_logs` evidence before archiving/removing the branch/worktree; do not promote these historical producers into the active tool surface.  Rebuild a future launch gate against the then-current outlet and identity contracts. |
| `codex/schema28-core-recovery-pen-20260903` | No code admission remains.  Commit `0973f19` is not whole-commit patch-equivalent, but 13 of its 24 per-file patches—including `dynamics.py`, `intent.py` and `types.py`—are stable-patch identical in later mainline commit `cb7f0fd`; the remaining surfaces replay the recovery together with layout-2, optional B-spine and the stricter profile-owned gripper boundary.  Current manifest, single-forward training, profile-boundary, validation-boundary and resume-schema tests close that supersession.  Retain an immutable provenance tag for the formal recovery run before removing the redundant branch/worktree. |
| `codex/schema25-r1-replay` | Its sole branch commit `90557b6` is confirmed patch-equivalent to the integration line, so no code admission remains.  The worktree still owns modified/staged historical documents, two untracked candidate memos and local `.audit` scripts/results; classify those records independently before any removal. |
| `codex/schema28-estimator-gate` | No source or test admission remains.  Whole-commit patch identity differs because `6a43e8e` edited an intermediate repair-plan state, but its `logging.py`, `train.py`, architecture-contract and two test-file patches have the same stable patch IDs as mainline commit `ab80fe3`; the unmatched plan narration is superseded by this living plan.  Preserve the untracked `.audit/schema28_estimator_gate_6a43e8e_gate2` replay record as experiment provenance before archiving/removing the branch and worktree. |
| `codex/v94-latent-ownership-execution` | Keep while the protected stash depends on it. |
| `codex/v86-slot-controller` | Extract the small real source/test/launcher WIP from line-ending noise before removal. |
| `codex/v76-recovered` | Historical ancestor.  Verify its retained tag/provenance, then remove the branch. |
| `origin/codex/hybrid-v1` | Three unique commits form an optional hybrid Pen experiment and must not be merged wholesale.  The final `cd9489a` exposes a separable deployment-identity gap—the committed validator writes but does not validate `architecture_manifest`—but its patch is hybrid-schema-specific, has no focused negative test, and overlaps the root worktree's active `deployment.py`/`manifest.py` edits.  After that owner closes the WIP, either confirm the gap was superseded or extract a current-schema manifest-validation unit with explicit legacy compatibility tests.  Retain the remote ref until then. |
| `origin/codex/rdt-data-adaptation` | Remote ancestor and later deletion candidate; revalidate against the server before deletion. |

Before removing any worktree or branch, require: clean/understood status,
recoverable unique commits, preserved intentional untracked files, no dependent
stash, a recorded replacement ref where needed and an ancestry/patch-equivalence
check against the chosen `master`.

## Active work order

### 1. Close outlet contracts before changing the shared core

- RDT: treat the old recovery event metrics as incompatible; if RDT remains a
  release outlet, start one corrected new initialization only after the
  consolidated mainline source is fixed and identifiable.
- CALVIN: keep the shared value/adjacent-difference 18-D core.  Validate the
  outlet-scoped centered-command sampler, W projection and binary gripper
  isolation with a complete formal curve, read-only causal replay and
  closed-loop `open_drawer` result.
- Pen: retain the completed recovery curve as the shared-core guard.  Do not
  change S/W/P/outer closure merely because far/event behavior remains weak;
  first select the producer-to-consumer failure with matched evidence from the
  current issue ledger.

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

### B-spine

B-spine remains an optional `execution_bottom` experiment.  Do not name it a
new mainline schema or run a training sweep until the repository has one clean
trunk, the active outlet contracts are closed and the user explicitly selects
the experiment.  If resumed, use only this compact ladder:

1. one read-only cross-outlet chart/locality probe;
2. one opt-in bottom implementation with the original full-resolution path
   retained;
3. focused forward/reverse, checkpoint, real CUDA and lifecycle gates;
4. one matched Pen pair, rejecting gains caused by event/detail suppression.

No learned knots, output-space spline writer, extra ODE/W loop, hand-tuned
gain, quota or smoothing objective belongs to that first experiment.

### Benchmark and simulation

The untracked benchmark/simulation work is tooling until it has independent
tests, explicit external-dependency boundaries and a canonical launcher.  It
must not be used to claim mainline behavior or be bundled with action/outlet
semantics.  Remove direct legacy-lab dependencies or label them as optional
compatibility before promotion.

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
   from the accepted default until its own gates close.

At each closure, update the architecture contract only when accepted behavior
actually changes, update the issue ledger in place, and rely on Git history
rather than rebuilding a chronological diary in this file.
