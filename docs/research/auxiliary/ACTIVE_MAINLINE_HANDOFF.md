# ClearVLA active operational handoff

Snapshot: 2026-09-06, canonical namespace and live import verified on `senwang-server`.

This file is the single path contract and volatile process map. It does not
define model architecture. Recheck each PID, `/proc/PID/cwd`, output target and
serialized `run_context.json` before acting. Architecture lives in
[../00_CURRENT_ARCHITECTURE_CONTRACT.md](../00_CURRENT_ARCHITECTURE_CONTRACT.md);
open decisions live in
[../CURRENT_MAINLINE_ISSUES.md](../CURRENT_MAINLINE_ISSUES.md).

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

For the three already-running jobs these entries are links to the
existing physical files and checkout; no live source, output, log or checkpoint
is moved or deleted. Resolve `console.log` from the target of
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

| Status | Job | Parent PID | Canonical experiment view | Physical checkout/output |
|---|---|---:|---|---|
| current Pen | Pen raw-W q5 | 2549150 | `experiments/pen/20260906-ws-p2-q5` | `/home/sen.wang/workspace/robotics/clear/pen-core-q5-20260906`; `runs/pen_raw_ws_p2_q5_5dbe5ee_20260906_formal` |
| current CALVIN | CALVIN ABC-D selective | 613786 | `experiments/calvin/20260906-abc-d-selective` | `/home/sen.wang/workspace/robotics/clear/clearvla_sim_mainline`; `/data/senwang/data/calvin/runs/clearvla_calvin_abc_d_expanded_selective_v1_20260906` |
| running historical | Pen hybrid v1 | 4030788 | `experiments/pen/20260905-hybrid` | `/home/sen.wang/workspace/robotics/clear/hybrid-v1-pen-20260905`; `runs/hybrid_v1_pen_b8_formal_final_20260905` |

The outlet pointers are:

```text
active/pen    -> ../experiments/pen/20260906-ws-p2-q5
active/calvin -> ../experiments/calvin/20260906-abc-d-selective
```

The hybrid job remains protected while it runs, but it is historical and must
not become `active/pen`. Preserve every DataLoader worker belonging to the
three parent processes. Do not edit their checkouts or clean their outputs and
caches while active.

## Existing retained runs

Five completed run trees, about 27.03 GB total, remain physically under
`/data/senwang/archive/clearvla-runs/2026-09-06/`. Their old checkout-local
`runs` paths are compatibility symlinks:

```text
schema31-bspine-pen-20260904
schema31-arm-only-pen-20260905
schema28-core-recovery-rdt8-formal-20260904
schema28-core-recovery-pen-20260903
schema29-rdt-multitask
```

Expose retained evidence through the canonical `archive/` namespace without
copying it again. Two inactive environments previously removed from
`rdt-multitask-prep/.venv` and `clearvla_v112_fix_belongings/.venv` can be
recreated from their `pyproject.toml` and `uv.lock`.

## Transition and cleanup boundary

- Import an existing active run as a link-only view; never relocate it during
  execution.
- Create future runs directly in `experiments/` and select a tested fixed
  checkout from `checkouts/`.
- Keep one human-facing namespace. Documentation must link here instead of
  publishing another current-path table.
- Treat dirty checkouts, unknown ownership, datasets, caches used by an active
  process, formal checkpoints and serialized contexts as retained state.
- Before any archive or cleanup, verify process identity, checkout status,
  latest file time and destination capacity. Preserve old absolute paths when
  compatibility requires them.
- Keep raw logs and binary artifacts out of Git; document only identity,
  decisions, source references and reproducible audit information.
