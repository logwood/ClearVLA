# ClearVLA simulation and benchmark runbook

This is a development runbook for simulator data and external evaluators. It
does not define the G/S/W/P architecture, live process state, or benchmark
results. Read the
[`architecture contract`](../research/00_CURRENT_ARCHITECTURE_CONTRACT.md)
for model semantics and the
[`rolling handoff`](../research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md) for
current remote jobs.

## Supported lanes

| Lane | Role | Status |
|---|---|---|
| `alicia-proxy` | MuJoCo camera/control smoke with an Alicia-shaped proxy | diagnostic only |
| `maniskill-stackcube` | ManiSkill 3 `StackCube-v1` data/control smoke | native simulator lane |
| CALVIN ABC-to-D | external evaluator and converted data outlet | primary external benchmark |
| LIBERO | external evaluator and converted data outlet | secondary comparison |

The simulator policy sees only the admitted observation contract:

```text
top/wrist RGB + measured 7D state + previous executed 7D action
  -> ClearVLA history
  -> predicted 24-step chunk
  -> execute first action only
  -> simulator step
  -> evaluator-only reward/success
  -> atomic HDF5 episode + manifest
```

Object poses, goal coordinates, reward and success never enter the policy
input. Random/hold/extreme episodes prove protocol closure only; they are not
expert data.

## Data contract

An opt-in [residual SAC pilot](../../clearvla/rl/README.md) now has its own
ManiSkill base profile, reward/replay learner and preparation gates. Its reward
may supervise the external learner but never enters the online observation.
It is not enabled by ordinary simulator/evaluator or mainline training commands.
New StackCube base preparation uses the v2 profile: causal reset history,
48 real tail steps, source-center bounds including immediate release, and the
fixed-down-reference causal rotation chart. The residual runner selects this
chart explicitly; generic simulator rollout requires
`--maniskill-state-chart fixed_down_causal_rotvec_v2` for a v2 checkpoint and
rejects mismatched state semantics. Legacy runs remain in their own namespace.

ManiSkill uses normalized `pd_ee_delta_pose` actions:

- state: TCP xyz, rotation-vector orientation, finger opening;
- action: native normalized 7D command;
- `action_state`: previous clipped native action;
- minimum accepted expert episode: 58 frames, replayed success, bounded actions,
  `action[t] == action_state[t+1]`, and a passing ClearVLA loader audit.

Simulator and benchmark roots must remain separate from Pen/RDT roots. Each
outlet owns its own normalizer, decoded cache, DINO cache, language bank,
configuration and checkpoint. Never mix converted smoke data with expert data.

CALVIN conversion keeps terminal action tails rather than truncating windows:
policy centers through `terminal-24` are followed by 24 absorbing frames
(terminal RGB/state repeat, arm command zero, gripper command held). Synthetic
rows are excluded from normalizer fitting, and source-trajectory split identity
is preserved.

LIBERO has a separate outlet chart.  Its converter admits native seven-column
normalized `OSC_POSE` actions, EEF xyz/axis-angle plus opening-width state, and
writes `action_state[0] = 0` followed by the previous executed action.  The
official filename-derived `Task.language` is the policy instruction; a BDDL
`:language` declaration is retained only as simulator provenance.  Conversion
requires the official BDDL task inventory so this mapping is verified rather
than inferred from HDF5 metadata alone.  It audits every episode in a staging
root before an atomic publish, so a malformed task cannot leave a partial
converted root.  The evaluator applies fixed init states, ignores success/done
during the five zero-action physics warmup, then
executes only the first row of each 24-row policy chunk after `[-1,1]` clipping;
that clipped row is fed back as the next action-state.  These checks establish
edge protocol closure, not a learned LIBERO score.  Fixed states are consumed
in the official first-N order and are never silently recycled.  Evaluate
`libero_90` and `libero_10` separately: `libero_100` is their archive union,
not an official evaluator suite.  A result is marked `official_protocol` only
for a formal checkpoint bridge over every suite task with 20 fixed states, 600
policy-action steps, the five-step warmup, seed 10000 and 128-pixel camera
inputs; a smoke-zero run is always marked non-official even when its counts
match.  The result identity hashes each used first-N init-state matrix and BDDL
file and records `image_side`; changing any of them is not a valid resume.
The bridge endpoint is operational provenance and may move when the immutable
checkpoint/graph/profile/normalizer identity is unchanged.  Per-task atomic
files remain `complete: false`; only the final bridge-identity check publishes
`complete: true`, so an interrupted aggregate cannot be mistaken for a score.

## Remote setup

Use an isolated checkout and environment on the Linux host. The setup scripts
own the exact paths and optional dependency installation:

```bash
bash scripts/setup_remote_simulation.sh
bash scripts/smoke_remote_simulation.sh
```

For benchmark evaluator environments:

```bash
bash scripts/setup_remote_benchmarks.sh calvin libero
bash scripts/smoke_remote_benchmarks.sh calvin libero
```

Keep CALVIN/LIBERO evaluator environments separate from the Python 3.12
ClearVLA converter environment. On Windows, use the Linux host for these
legacy MuJoCo evaluators.

### CALVIN inline evaluation video

Formal and targeted CALVIN evaluation can record the observations from the
same scored environment execution. Recording is off by default and does not
replay actions or invoke the policy a second time:

```bash
PYTHONPATH=/home/sen.wang/workspace/robotics/clear/clearvla_sim_mainline \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl QT_QPA_PLATFORM=offscreen \
/home/sen.wang/.venvs/clearvla-calvin/bin/python \
  -m clearvla.benchmarks.calvin_eval \
  --dataset-root /data/senwang/data/calvin/raw/task_ABC_D \
  --output-dir /data/senwang/data/calvin/rollouts/FORMAL_RUN \
  --endpoint http://127.0.0.1:18772 \
  --num-sequences 16 --max-subtask-steps 180 --execute-rows 4 \
  --record-video --video-fps 12 --video-sequences 16
```

Unless `--video-dir` is provided, each completed official sequence is flushed
immediately as `OUTPUT_DIR/videos/sequence_NNNN.mp4` with a matching JSON task
report. `--video-sequences N` limits recording to the first N evaluated
sequences; omit it to record all of them. An interrupted current sequence keeps
a `.partial` artifact, while videos from earlier completed sequences remain
usable.

## Minimal smoke and import commands

One bounded simulator rollout:

```bash
/data/senwang/envs/clearvla-sim/bin/python -m clearvla.simulation.rollout \
  --environment maniskill-stackcube \
  --policy random --seed 1 --steps 200 --max-episode-steps 200 \
  --record-dir /data/senwang/data/clearvla_sim/maniskill_stackcube_v1 \
  --episode-id random_000001
```

Audit a converted or recorded root:

```bash
/data/senwang/envs/clearvla-sim/bin/python -m clearvla.simulation.dataset \
  /data/senwang/data/clearvla_sim/maniskill_stackcube_v1
```

Import official StackCube demonstrations only through the checked converter:

```bash
bash scripts/import_maniskill_stackcube_demos.sh
```

The importer must re-execute the converted actions, verify final success and
causal alignment, and write expert output under a new namespace.

## Benchmark conversion

First use bounded protocol roots; never promote a bounded root in place:

```bash
CLEARVLA_CALVIN_LIMIT_PER_SPLIT=2 \
  CLEARVLA_LIBERO_LIMIT_TASKS=1 \
  CLEARVLA_LIBERO_LIMIT_DEMOS=3 \
  bash scripts/convert_remote_benchmarks.sh calvin-debug libero-spatial
```

For a full conversion, set a new output namespace for converted data, caches,
language bank, config and runs. The scripts expose
`CLEARVLA_*_CONVERTED`, `*_CACHE_ROOT`, `*_LANGUAGE_BANK`,
`*_CONFIG` and `*_RUN_ROOT` for this purpose.

Build independent language/vision artifacts after conversion:

```bash
bash scripts/build_remote_benchmark_artifacts.sh calvin --language --caches
bash scripts/build_remote_benchmark_artifacts.sh libero --language --caches
```

Audit ingress and typing before training:

```bash
/data/senwang/envs/clearvla-sim/bin/python -m clearvla.benchmarks.audit \
  /data/senwang/data/libero/converted/libero_spatial_full

/data/senwang/envs/clearvla-sim/bin/python -m clearvla.benchmarks.training_smoke \
  --config /data/senwang/data/libero/configs/libero_spatial_full.json \
  --output /data/senwang/data/libero/validation/libero_spatial_full_training_smoke.json
```

These are data/typing checks, not policy scores. A missing language bank or
failed visual-parity preflight is a preparation failure, not a reason to reuse
the Pen condition or another outlet's cache.

## Raw-reader boundary

The read-only CALVIN raw reader is a parity/source interface, not the formal
training path:

```bash
/data/senwang/envs/clearvla-sim/bin/python -m clearvla.benchmarks.calvin_raw \
  --source /data/senwang/data/calvin/raw/task_ABC_D \
  --task-filter open_drawer
```

It must preserve trajectory inventory, annotations, 24/48-frame causal windows,
absorbing-terminal rules and row-level parity with converted HDF5. Formal runs
continue to use the converted HDF5/cache interface until raw image/token
throughput and memory have been measured.

## Safety rules

- Do not overwrite an older checkout, bounded conversion root or output root.
- Keep simulator, CALVIN and LIBERO normalizers/caches/checkpoints separate.
- Do not treat random, hold, extreme or training-smoke success as benchmark
  performance.
- Keep generated images, HDF5, caches, language banks and checkpoints outside
  Git, normally below `/data/senwang/data/`.
- A formal benchmark requires a validated checkpoint and the official evaluator;
  smoke only proves the environment-to-observation bridge.
