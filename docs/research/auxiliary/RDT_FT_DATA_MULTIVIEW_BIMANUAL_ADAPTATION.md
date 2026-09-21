# RDT-8 data and adapter contract

Updated: 2026-09-05

This file defines the bounded RDT-8 outlet. It does not claim native
three-camera, depth or 14-D bimanual support. Shared model semantics remain in
the [architecture contract](../00_CURRENT_ARCHITECTURE_CONTRACT.md); live jobs
remain in [the handoff](ACTIVE_MAINLINE_HANDOFF.md).

## Supported outlet

```text
source:           /data/rdt-ft-data
tasks:            8
model cameras:    high, right_wrist
native source:    action/qpos [T,14]
model action:     right-arm indices 7..13 -> [T,7]
profile:          rdt_right_arm_action_chart_v1
horizon:          action 24 / world 48
language:         original HDF5 instruction + T5-v1.1-XXL bank
split:            143 / 18 / 18 selected episodes
windows:          54,648 / 6,711 / 6,990 train/val/test
gripper boundary: previous executed command
```

This is a two-camera, right-arm, seven-dimensional multitask adapter. The
source corpus contains additional modalities and coordinates that are not
consumed by the model.

Canonical files:

- [data config](../../../configs/mainline/rdt_multitask8_data_v1.json)
- [selection spec](../../../configs/mainline/rdt_multitask8_selection_spec_v1.json)
- preparation: `scripts/prepare_rdt_ft_data.sh`
- model smoke/train: `scripts/smoke_rdt_multitask.sh`,
  `scripts/train_rdt_multitask.sh`

## Identity and language

Episode identity is `source_partition/task/episode`, suffix-free and
root-relative. The same identity owns split membership, cache addresses,
language mapping and inventory. The manifest fails closed on duplicates,
missing/unknown partitions, changed discovery or stale policy.

The selected tasks are:

`put_cherry_bowl, stack_tomato_cans, draw_triangle, wipe_glass_water,
press_stapler, grab_stick_into_bottle, shake_glass, pour_water_can2cup`.

One train-only normalizer covers all tasks. Task identity is sampling,
validation and logging metadata only; it is never a model condition.

The original HDF5 instruction is the only training language distribution. The
bank retains the first 32 T5-v1.1-XXL tokens and a real attention mask; exact
instruction text is the lookup key. Missing or stale mappings fail before
training, and Pen language cannot be substituted.

## Camera and tensor contract

Camera names are ordered and unique. A three-view cache may serve the selected
two-camera tuple only through explicit indices. Camera count or reference
encoder batch shape is part of deployment ABI.

```text
online DINO:  [B,3,2,256,768]
future DINO:  [B,12,2,256,768]
online RGB:   [B,3,2,3,336,336]
goal tokens:  [B,32,4096]
action:       [B,24,7]
future state: [B,48,7]
history:      [B,8,7]
```

Missing depth is not an observed zero. Depth and a third camera require a new
ABI, memory gate and fresh checkpoint.

## Action and gripper chart

Native action is the command/target chart; qpos is the observed state chart.
The right gripper command conversion uses the profile-owned scale
`13.9231 / 4.7888`; this is a chart coefficient, not a physical limit.
RDT units and mechanical limits remain `source_native` until independently
verified.

The previous executed command owns row zero and the full continuous-gripper
boundary for encode/decode, adjacent deltas, sampling, loss and evaluation.
Partial row-zero substitution is invalid. The pinned adjacent-command threshold
is `0.18310546875`; it is not Pen's threshold and cannot be overridden by
a formal launcher.

## Sampling and validation

At batch size eight, task-first sampling provides one row per task before
uniform/event/motion lanes. Validation reports equal task counts plus native and
normalized full/arm/gripper RMSE, horizon bands, decoded event P/R/F1 and both
micro and equal-task macro aggregates. An absent task is reported absent.

Preparation/data smoke and model smoke establish data/typing or interface/
gradient health only. Neither is a behavior result.

```bash
RDT_PREPARE_THROUGH=manifest bash scripts/prepare_rdt_ft_data.sh
RDT_PREPARE_THROUGH=language bash scripts/prepare_rdt_ft_data.sh

RDT_SMOKE_SPLIT=val RDT_SMOKE_EPISODE_LIMIT=1 \
  bash scripts/prepare_rdt_ft_data_bounded_smoke.sh

bash scripts/smoke_rdt_ft_data.sh

OUT_DIR=runs/rdt8_smoke_TIMESTAMP bash scripts/smoke_rdt_multitask.sh
RDT_MAX_VAL_BATCHES=64 OUT_DIR=runs/rdt8_new_TIMESTAMP \
  bash scripts/train_rdt_multitask.sh
```

Generated caches, language tensors, reports and checkpoints stay outside Git.

## Deferred expansion

- **Three RGB cameras:** preserve the camera axis through G, Teacher, W,
  P2 and transition; current dimensions require two cameras.
- **Native 14-D bimanual action:** requires explicit two-limb value/delta and
  gripper fields, objectives, decode, checkpoint and per-limb evaluation.
  Changing `action_dim` alone is not support.
- **Depth:** requires TIFF units, invalid-pixel semantics and an availability
  mask.

Each expansion is a separate source unit. It must pass typed forward/backward,
optimizer, checkpoint, deployment, memory and complete per-task behavior gates.
