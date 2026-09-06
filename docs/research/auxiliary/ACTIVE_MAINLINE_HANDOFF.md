# ClearVLA active operational handoff

Snapshot: 2026-09-06 20:11 +08:00 on `senwang-server`.

This file is a volatile process and storage map, not an architecture record.
Recheck every PID, checkout, output path and serialized run context before
acting. Architecture lives in
[../00_CURRENT_ARCHITECTURE_CONTRACT.md](../00_CURRENT_ARCHITECTURE_CONTRACT.md);
open decisions live in
[../CURRENT_MAINLINE_ISSUES.md](../CURRENT_MAINLINE_ISSUES.md).

## Protected active jobs

| Job | Parent PID | Checkout | Output |
|---|---:|---|---|
| CALVIN ABC-D selective | 613786 | `/home/sen.wang/workspace/robotics/clear/clearvla_sim_mainline` | `/data/senwang/data/calvin/runs/clearvla_calvin_abc_d_expanded_selective_v1_20260906` |
| Pen hybrid v1 | 4030788 | `/home/sen.wang/workspace/robotics/clear/hybrid-v1-pen-20260905` | `runs/hybrid_v1_pen_b8_formal_final_20260905` |
| Pen raw-W q5 | 2549150 | `/home/sen.wang/workspace/robotics/clear/pen-core-q5-20260906` | `runs/pen_raw_ws_p2_q5_5dbe5ee_20260906_formal` |

The corresponding DataLoader workers are children of these parent processes.
Do not interrupt them, edit their checkout, or clean their output/cache while
they are active. Identify a job from both its command line and `/proc/PID/cwd`.

The CALVIN run uses
`configs/mainline/calvin_abc_d_expanded_selective_v1.json`; the hybrid run uses
`configs/mainline/object_intent_dynamics_323_pen_hybrid_v1.json`; the q5 run
uses `configs/mainline/object_intent_dynamics_323_pen_w_interval_q5.json`.
Read each `run_context.json` before comparing or resuming it.

## Retired process records

PIDs `919391`, `2344176`, `2972941`, `618348`, and `618376` were absent at this
snapshot. They are historical process identities, not protected active jobs.
Their evidence remains in run directories, local `new_logs/`, and Git history.

## Storage layout

Heavy completed runs belong on `/data`; checkout-local `runs` paths may be
symlinks so old commands and references continue to resolve. The following
completed runs were moved without deleting checkpoints or metrics:

```text
/home/sen.wang/workspace/robotics/clear/schema31-bspine-pen-20260904/runs
/home/sen.wang/workspace/robotics/clear/schema31-arm-only-pen-20260905/runs
/home/sen.wang/workspace/robotics/clear/schema28-core-recovery-rdt8-formal-20260904/runs
/home/sen.wang/workspace/robotics/clear/schema28-core-recovery-pen-20260903/runs
/home/sen.wang/workspace/robotics/clear/schema29-rdt-multitask/runs
```

All five resolve under
`/data/senwang/archive/clearvla-runs/2026-09-06/` and total about 27.03 GB.
Two inactive, lockfile-rebuildable environments were removed from
`rdt-multitask-prep/.venv` and `clearvla_v112_fix_belongings/.venv`, releasing
about 5.68 GB. Recreate either environment from its `pyproject.toml` and
`uv.lock` if that checkout is reused.

## Cleanup boundary

- Never delete a broad workspace, data root, active run, dataset, formal
  checkpoint, serialized context, or cache used by an active process.
- Before moving a completed run, verify the parent PID is absent, inspect its
  checkout status and latest file time, then preserve the old absolute path
  with a symlink.
- Treat dirty checkouts and unknown ownership as retained state, even when
  their run directories are large.
- Keep raw logs out of Git. Retain decision-making summaries and reproducible
  audit commands in documentation.
- Recheck remote disk space and process identity before every cleanup batch.

Useful read-only checks:

```bash
df -h / /data
ps -eo pid,ppid,stat,etimes,args | grep -E 'clearvla|torchrun|python'
readlink -f /proc/PID/cwd
du -x -d 1 -B 1 /home/sen.wang/workspace/robotics/clear | sort -n
```
