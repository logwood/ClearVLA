# ClearVLA Schema30 rolling handoff

Snapshot: 2026-09-02 13:14 +08:00

This file is the volatile operational snapshot. It does not define
architecture; read
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md)
first, then
[`../CURRENT_MAINLINE_ISSUES.md`](../CURRENT_MAINLINE_ISSUES.md) and
[`../CURRENT_MAINLINE_REPAIR_PLAN.md`](../CURRENT_MAINLINE_REPAIR_PLAN.md).
Recheck remote state before acting because PIDs and steps below will age.

## Identity

```text
capability:             object_intent_dynamics_323
manifest schema:        30
manifest digest:        1323dcff095cbddb8da02c0e263c3e9865fbae39add9af4e539d38e9745f9c46
Schema30 source commit: 3fef2fc0dce297f600c813307c998f587cca1ca3
release-doc checkout:   f60bd808becabd882b10ad7b07e74242fe49a881
Linux source digest:    0d0957a75ab22e37f552ccf9a4505049876af5785837cb9787edde181b04c1c2
local branch name:      codex/schema29-mainline (historical name only)
```

The remote formal workspace is a detached checkout of `f60bd80`:

```text
/home/sen.wang/workspace/robotics/clear/schema30-mainline-3fef2fc
```

Its only observed untracked path is `diagnostics/`; active source identity is
serialized in each run context. Branch and directory names never override the
manifest or source digest.

## Release gates already passed

- local regression/static: 223 passed, 2 CUDA-only skipped; changed-file Ruff
  and compileall passed;
- fresh checkpoint save/load and Schema29 exact-resume rejection;
- real Pen B8 CUDA BF16 cache0/cache1 parameter VJP;
- fresh Pen B8 and task-balanced RDT-8 smokes with finite backward, exact loss
  ledger, deploy-style validation and atomic checkpoints;
- read-only checkpoint validation for both outlets with
  `source_delta_files=0`, no optimizer/scheduler/RNG load and no write.

These are release/interface gates only. They do not establish learned
performance.

## Active formal runs

### Pen core-behavior outlet

```text
GPU / PID:    0 / 2004608
run tag:      schema30_pen_b8_20260902_115644
run dir:      runs/schema30_pen_b8_20260902_115644
console log:  schema30_pen_b8_20260902_115644.log
config:       configs/mainline/object_intent_dynamics_323.json
batch/workers: 8 / 4
```

At the snapshot, the process was alive and the last completed compact window
was epoch 1 step 2000. No validation row had completed. Ledger and finite-value
scans were clean. Seven threshold-5 crossings occurred through step 106, all
dominated by the output head; no further Pen crossing appeared through step
2000. This is early health evidence, not a behavior result.

### RDT-8 adapter/multitask outlet

```text
GPU / PID:    1 / 2005400
run tag:      schema30_rdt8_b8_20260902_115726
run dir:      runs/schema30_rdt8_b8_20260902_115726
console log:  schema30_rdt8_b8_20260902_115726.log
config:       configs/mainline/rdt_multitask8_data_v1.json
batch/workers: 8 / 4
validation:   at most 64 batches per task-facing panel
```

At the snapshot, the process was alive and the last completed compact window
was step 1500. No validation row had completed. The first three finite
crossings were output-head events. Three later observation-side crossings
appeared at steps 954/1251/1317: the maximum was
`target_dino_key.1.weight` owner L2 `19.28`, global preclip `22.60`, with
later ownership split between `target_dino_key` and `flow.delta_head`.
There was no non-finite, traceback, lineage or ledger failure. This recurrence
deserves checkpoint/validation review but does not yet prove the proposed
address-chain root cause.

The earlier `schema30_pen_b8_20260902_115507` directory is a zero-step launch
failure caused by a non-interactive PATH that could not locate Python. It
contains no training or checkpoint evidence.

## Immediate next action

1. Verify both PIDs and read the newest compact train/validation rows.
2. Continue both runs unless a hard-stop condition in the current plan fires.
3. At the first completed validation, audit the run directory rather than only
   the console log.
4. Compare Pen with the complete Schema28 anchor and RDT-8 task by task.
5. Do not edit the graph from early train loss, event F1, geometry magnitude or
   one finite spike.

Useful read-only commands on the server:

```bash
cd /home/sen.wang/workspace/robotics/clear/schema30-mainline-3fef2fc
ps -p 2004608,2005400 -o pid,etimes,stat,%cpu,%mem,cmd
tail -n 80 schema30_pen_b8_20260902_115644.log
tail -n 80 schema30_rdt8_b8_20260902_115726.log
python -m clearvla.tools.audit_policy_logs +  runs/schema30_pen_b8_20260902_115644 --format text
```

For RDT-8, pass its run directory to the same auditor and inspect all per-task
rows. A compact aggregate is insufficient.

## Checkpoint and continuation boundary

- Both formal runs started fresh in new directories.
- Schema29 and earlier checkpoints are not Schema30 exact-resume or migration
  inputs.
- Smoke checkpoints are gate artifacts, not formal initialization sources.
- Resume only from a checkpoint whose run context matches the complete
  Schema30 source/config/manifest/data/optimizer identity.
- Validation-only checkpoint loading must remain read-only.

## Local documentation state

The local branch is `codex/schema29-mainline`. Its current HEAD contains the
documentation compaction on top of formal-run checkout `f60bd80`: active
truth is shortened, replay/design evidence is archived, legacy README files
are consolidated, and raw logs are excluded from Git while retained on disk.
Verify the exact HEAD with `git rev-parse HEAD` rather than copying a stale
hash into this rolling file.

Separate CALVIN/data-adapter source and test edits may remain unstaged in the
same worktree. They are not part of the documentation commit. Preserve and
review them independently; do not reset or accidentally fold them into a
documentation-only change.

The pre-compaction documents and retired R1/R2 worksheets remain recoverable at
`f60bd80`. No checkpoint, tensor cache or raw probe dump should be added to
the documentation commit.

## Historical retrieval

- compact R1/R2 decisions:
  [`R1_R2_CLOSURE_INDEX.md`](R1_R2_CLOSURE_INDEX.md);
- replay provenance:
  [`../archive/replay/`](../archive/replay/README.md);
- older research evidence:
  [`../archive/legacy_evidence/`](../archive/legacy_evidence/README.md);
- RDT interface boundary:
  [`RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md`](RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md).

Open those only for ancestry, an old log or the reason for an earlier repair.
Never reconstruct the current graph from a historical experiment name.
