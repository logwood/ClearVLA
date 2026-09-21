# ClearVLA active operational handoff

Updated: 2026-09-19 UTC. **Dated observations, not a live process inventory.**
This edit consolidates retained local evidence; it did not poll the server.
Each row below states when it was last checked. A documentation update does
not refresh a PID, checkpoint or training result.

## Scope and authority

The applicable user instruction defines the current task's scope. A historical
restriction belongs to its named task and experiment; it is not a permanent
repository-wide ban and does not revoke a later explicit user request. This
handoff is also not an independent authorization to launch, stop or modify work.
Architecture belongs in the [contract](../00_CURRENT_ARCHITECTURE_CONTRACT.md),
source/behavior status in the [issue ledger](../CURRENT_MAINLINE_ISSUES.md), and
proposed work in the [repair plan](../CURRENT_MAINLINE_REPAIR_PLAN.md).

| Task / owner | Current recorded scope | Boundary |
|---|---|---|
| Documentation cleanup — 增强执行者 | User requested correction of existing architecture/status documents | Documentation only; no model, source checkout or experiment changes |
| Old CALVIN V1 E4 evaluation — 增强执行者 | The authorized 34 bounded closed-loop trials are complete: 34/34 artifacts, 6/34 official successes; the final sync and local reconstruction are recorded below | The frozen panel is evidence only. It does not authorize stopping the old trainer, expanding the panel or starting another training run |
| DataLoader integration — 增强执行者 / 新主执行者 | Coordination completed; first candidate is optional pread on the existing NPY layout, legacy default | No production activation recorded; preserve seeded sample/batch/RNG and tensor identity |
| B1 camera / 24-row W sequence — 新主执行者 | Source implementation and its separately scoped training/selection work belong to task `01a09943-12b5-7342-8b9a-d398de7e3a6a` | Do not edit its active source, scripts or processes from this task; direct coordination is already authorized |
| Final structural evidence round — closed | Existing causal-ladder artifacts and B1 initialization review are complete | Do not revive that diagnostic round or the old two-5000-step pilot automatically; this restriction does not prohibit later separately authorized evaluation |

## Last verified observations

These are retained snapshots. Recheck the user's process identity, cwd, stdout
and source before an operational action; do not treat an old PID as authority.
Older Pen/RDT/LIBERO/StackCube observations were not refreshed in this cleanup.

| Last verified (UTC) | Run / source | Observed state | Recorded parent PIDs |
|---|---|---|---|
| 2026-09-19 17:16 | V1 E4 panel / `34ee78e` | Final 34/34 cases complete; official success 6/34; local artifact reconstruction and source SHA256 comparison passed. Push 2/18, target contact 11/18 (left 9/9, right 2/9). Controller children were reaped. | Controller `3366272`, bridge `3366277`, evaluator `3370260` |
| 2026-09-19 06:16:30 | V1 E4 panel / `34ee78e` | First real action executed, case 0 / step 1; video recording opened; full panel not complete at that observation | Controller `3366272`, bridge `3366277`, evaluator `3370260` |
| 2026-09-19 05:23 | Old V1 training / `34ee78e` | E5 / step 117958; saved E4 / 105308 was available for the independent panel | Trainer `3723394` |
| 2026-09-19 05:23 | Old V2 training / `635213d` | E1 / step 16950 of 26327; no completed checkpoint in the audited inventory | Trainer `76469`, launcher `72610` |
| 2026-09-19 05:23 | Camera arm / `aa798d12` | This run E1 / 4000 of 11012; no completed validation/checkpoint in the audited inventory | Trainer `2111551` |
| 2026-09-19 05:23 | Sequence arm / `17a06a76` | This run E1 / 4000 of 11012; no completed validation/checkpoint in the audited inventory | Trainer `2111533` |

The historical old V1/V2 and six-task Q5 arms differ in data/normalizers.
Neither their losses nor these progress rows establish a matched architecture
comparison. Camera/sequence initialization from E1 starts fresh optimizer,
schedule and RNG; it is a new training run, not an exact resume.

The final V1 E4 panel is now a closed behavioral baseline, not a live process
status. Its local canonical copy is
`artifacts/calvin_e4_20260919_download/`; the rebuilt case-level analysis is in
`analysis/summary.md`, `analysis/statistics.json`, `analysis/cases.jsonl` and
`analysis/inventory.json`. The local rebuild verified all 34 case files and
flat videos against the untouched final sync, and verified `steps + 7` video
frame timing. The panel remains 34 independently reset trials, not the
official 1000-chain benchmark.

Snapshot evidence: `artifacts/closed_loop_selection_20260919/REVIEW.md`,
`run_evidence.json`, `v1_checkpoint_metadata.json` and the evaluation's
`startup_verification.json`. These are local/server evidence locations; a fresh
Git checkout may not contain the ignored artifacts.

## Authorized V1 E4 panel and file locations

Remote root:

```text
/data/senwang/clearvla/experiments/calvin/20260919-object-binding-v1-e4-comprehensive
```

Frozen checkpoint: E4 / 105308, SHA-256
`327e4a3d1a946568330b76c6b3fc5f57e67c2aa824d6c1c927759d47734d7119`.
The original trainer's `best.pt` may change without changing this copy.
The model uses its original direct-command ABI and original XVLA runtime.
An isolated entry restores only its serialized training-sidecar path for
config admission; no sidecar labels enter inference and all graph, weight,
source, normalizer and language checks remain active. Details are in that
run's `deployment_compatibility.json` and local `PLAN.md`.

| Output | Relative path under the remote root |
|---|---|
| Current controller status | `controller_status.json` |
| Current case / step | `shard_0/progress.json` |
| Finished videos | `shard_0/<case_id>_<task>_t<trial>/video/*.mp4` |
| Per-trial result / recorded trajectory | `shard_0/<case>/result.json`, `trajectory.npz` |
| Final all-trial results | `summary.json`, created only after all 34 cases complete |
| Startup / evaluator logs | `logs/bridge_0.log`, `logs/evaluator_0.log` |

Local canonical mirror and analysis (Windows workspace):

```text
C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts\artifacts\calvin_e4_20260919_download\data
C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts\artifacts\calvin_e4_20260919_download\videos
C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts\artifacts\calvin_e4_20260919_download\analysis
```

The final analysis keeps official oracle labels, computed trajectory flags and
older human video annotations in separate fields. It explicitly records that
rollout behavior cannot by itself localize a defect to G, S, W, P2 or the
bottom action generator.

Video files use hidden `.partial.mp4` names until a trial completes. A recording
file or successful first action does not establish a successful trial.
The bounded panel has 14 tasks, at most 360 steps per trial, one-row execution
and per-trial seed/history reset; it is not the official 1000-chain benchmark.
Local plan and startup evidence:
`artifacts/calvin_v1_e4_comprehensive_20260919/PLAN.md` and
`followup_state.json`. The user has received the output locations and requested
no active waiting for the entire panel. The existing periodic follow-up is
limited to meaningful failure/completion; it must not create new diagnostics,
repeat old probes or stop the old trainer. A stop/wait recommendation follows
results and is distinct from authorization to stop a process.

## DataLoader integration milestone

Coordination was recorded at 2026-09-19 05:19 UTC in
`artifacts/dataloader_impl_20260919/INTEGRATION_PLAN.md` and
`followup_state.json`. Implementation research is bounded and complete, with
production source unchanged. Proposed base `aa798d12...` and branch
`codex/calvin-loader-pread-20260919` were not created/activated by that
coordination. Revalidate the eventual selected model source's data interfaces.

Preparation belongs to this task; 新主执行者 owns final source review and
training admission. Production use waits for the agreed model-comparison
milestone and actual E1 normalizer/full-index/batch identity gates in the
[repair plan](../CURRENT_MAINLINE_REPAIR_PLAN.md#1b-integrate-the-exact-order-visual-reader-independently).
The completed six-episode I/O benchmarks do not establish end-to-end training
speedup or full-data resume equivalence. Phase-layout conversion and
compression remain separate proposals.

## Closed work and retained provenance

- The final structural evidence round is closed. Reuse
  `new_logs/current/20260918-target-y-shift-q5-v2/causal_ladder.json` and its
  tensor export; the independent review is
  `artifacts/shared_s_candidate_20260918/causal_ladder_audit.json`.
  Failed entity-to-K attribution and missing refined/terminal arrays remain
  stated evidence limits, not an automatic request for more probes.
- The actual Q5 reference was `1e51e288c700636ef7225b480c401db1603738ad`,
  E1 / 11012, checkpoint SHA-256
  `ac989d3c4e38feff106503aac0a0f74bfd8dad63ea20d551c58a7a5ef0ffdc17`.
  It already contained the shared-S query repair. Never substitute an
  overwritten/unverified `best.pt` or apply `635213d`'s missing-query finding.
- B1 source and restricted initialization review closed without an open source
  blocker. Earlier accepted P2/transition drift cases in the review JSON are
  pre-fix history. Source closure is not default promotion or behavior proof;
  details remain in `latest_followup_state.json` / `b1_final_review` under
  `artifacts/shared_s_candidate_20260918/` and in the issue ledger.
- The old `20260918-shared-s-preflight-goal-history` ended with exit 1 and zero
  batches/epochs; its converter-v2 terminal-padding gate rejected old episodes.
  Do not poll ended PIDs `121292/122015`, relax that data gate, or automatically
  revive `run_pilot_pair.py --launch`. No old 5000-step arm was started.

Earlier operation notes, including unpublished observations and the original
wording of task restrictions, are preserved in the
[historical handoff snapshot](../archive/operational_handoff_20260919.md).
That archive is for provenance only, not current commands or process state.

## Canonical remote namespace

The only human-facing ClearVLA root is:

```text
/data/senwang/clearvla/
├── repo/          fixed repository used to manage source
├── checkouts/     commit-specific Git worktrees
├── experiments/   canonical outlet/run views and future outputs
├── active/        one current formal-run pointer per outlet
└── archive/       retained inactive evidence
```

- `repo/` and `checkouts/<commit>/` own code. Do not create another full clone
  for every experiment or version label.
- `experiments/<outlet>/<short-run-id>/` is the canonical run location shown
  to people and tools.
- `active/<outlet>` is only a convenience pointer to the selected formal run.
  It is never a model-version, source or checkpoint identity.
- `archive/` retains inactive evidence. Archiving does not merge model
  versions or authorize checkpoint deletion.

Legacy `/home/sen.wang/workspace/robotics/clear/...` paths and existing
`/data/senwang/...` run roots remain physical storage details, not competing
human entry points.

## Experiment view contract

Each experiment exposes these stable data names (plus operational `entry.json`):

```text
experiments/<outlet>/<short-run-id>/
├── console.log
├── metrics.jsonl
├── run_context.json
├── checkpoints/
└── code
```

For imported historical runs these entries may be links to their
existing physical files and checkout; a view does not move a live source,
output, log or checkpoint. Resolve `console.log` from the target of
`/proc/PID/fd/1`; do not guess it from a run name. A view does not rewrite paths
serialized inside `run_context.json`. Treat the views as read-only: a symbolic
link is not a permission barrier, and writing through it changes the original.

Future formal launches write directly to their canonical `experiments/` run
directory and use a commit-specific worktree under `checkouts/`. Model and
resume identity still comes from manifest, resolved config, source digest,
data identity and `run_context.json`, never from the directory name.

## Commands

The root `workspace` entry points to the standard-library tool in the fixed
repository. It neither imports the model nor assumes a training Python path.

```bash
/data/senwang/clearvla/workspace list
/data/senwang/clearvla/workspace show --outlet pen --name 20260906-ws-p2-q5
tail -n 80 /data/senwang/clearvla/active/pen/console.log
```

For a **separately authorized future run**, first fetch the selected source
branch into `repo`, then pass its verified commit explicitly. Example argument
names below are placeholders, not a queued experiment:

```bash
git -C /data/senwang/clearvla/repo fetch origin "$SOURCE_BRANCH"
/data/senwang/clearvla/workspace checkout --ref "$VERIFIED_COMMIT"
/data/senwang/clearvla/workspace launch --outlet pen --name "$RUN_ID" \
  --ref "$VERIFIED_COMMIT" --config "$CONFIG_IN_COMMIT" \
  --python "$TRAINING_PYTHON" --gpu "$FREE_GPU_UUID"
```

`launch` runs in the foreground. For an explicitly requested detached run,
retain launcher stderr as well; do not discard it into `/dev/null`, because
an invalid ref/config/interpreter fails before the experiment is reserved.
The tool captures trainer output itself. During
startup stdout briefly occupies a uniquely named staging file beside the run;
after the trainer writes `run_context.json`, the same open file is atomically
published as `console.log` inside the run. Failed startup also retains its log
and exit code there. This preserves the trainer's empty-output-directory check.
An existing run ID or dirty checkout is rejected; no resume or migration is
implicit. `activate --outlet pen --name "$RUN_ID"` explicitly selects a verified
run and refuses to displace a running or unverified active entry. Launch does
not silently switch the user's active pointer.

`import-run --outlet ... --name ... --output ... --console ... --code ... --pid ...`
adds an existing run without moving it. It verifies context output, PID/cwd and
the live stdout path. `entry.json` records only operational paths/process identity;
it does not replace the model's `run_context.json` or save the shell environment.

## Protected active jobs

Protection follows verified ownership and the applicable user request, not
membership in an old PID table. This task has not stopped or modified the old
V1/V2 or the new main executor's experiments. Before acting on a process,
check that it still belongs to `sen.wang` and matches the intended source/run.
The old `active/pen` / `active/calvin` pointer targets in the historical snapshot
were not revalidated here and must not be advertised as current selections.

## Existing retained runs

Historical archive locations and inventories remain in the frozen handoff
snapshot. No run, checkpoint, environment or cache was moved or deleted by this
documentation cleanup; their current disk state was not checked.

## Transition and cleanup boundary

- Use Desktop Commander/SSH and the `senwang-server` skill for remote work.
  New small standalone `.sh` helpers belong in `/home/sen.wang/mysh`; read its
  `AGENTS.md` and existing scripts first. Repository-managed scripts stay in
  their repository.
- Import an existing active run as a link-only view; preserve its source,
  output, workers and caches. A convenience pointer is not model identity.
- Future formal runs use the canonical experiment namespace and an explicitly
  selected source. A documented example is not a queued launch or permission
  to change an active pointer.
- Keep raw logs, checkpoints, tensor caches and full probe dumps outside
  architecture memory. Retain concise decisions and reproducible references.
