# ClearVLA Current Policy Run

## Current Object Intent–Dynamics 3-2-3 Mainline

The active experiment is the capability-owned implementation under
`clearvla/mainline/`, selected by the serialized `object_intent_dynamics_323`
manifest. Its active architecture schema is 37; numeric V-labels and the old
monolithic `clearvla/policy/` graph are ancestry, not source-selection
mechanisms. Schema37 retains G1-G3, four future intervals, the V120 static P1,
the Evidence MMDiT/CVAE/workspace bottom and execution path while preserving
S/W/P2/P3 owner axes through their actual action-query consumers. The compact
executed graph and checkpoint policy are documented in
`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`.

Smoke (batch 1 by default; it executes backward and five-step deployment):

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema37_information_conservation_smoke \
nohup bash scripts/smoke_mainline.sh \
  > schema37_information_conservation_smoke.log 2>&1 &
```

Formal batch-eight run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema37_information_conservation_b8 \
nohup bash scripts/train_mainline.sh \
  > schema37_information_conservation_b8.log 2>&1 &
```

The raw HDF5 default remains
`/data/liang.zhang/dataset/grab_pen_single/grab_pen_single`. The current
launchers use `/data/senwang/data` only for decoded/DINO caches and
`/data/senwang/checkpoint` for T5/model weights. `DATA_ROOT`, `CACHE_DIR`,
`DINO_CACHE_DIR` and `T5_CONDITION_PATH` remain individually overridable.
Schema36 resume, optimizer resume and bottom-only migration are rejected;
Schema37 formal experiments start fresh in an absent or empty output directory.

Do not start the long run unless the batch-eight smoke remains below 22.0 GiB
total device use and completes backward plus five-step deployment. The
compact architecture and resource contract is in
`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`.

## Development Environment

The repository uses Python 3.12 and uv for dependency locking and development
tools. See `docs/development/uv_environment.md` for clean installation, reuse
of an existing CUDA Torch environment, and lightweight checks that do not run
the full policy trunk.

## Current Layout

当前实验主线位于 `clearvla/mainline/`；唯一默认训练与烟测入口是上一节的
`scripts/train_mainline.sh` 和 `scripts/smoke_mainline.sh`。`clearvla/policy/`
以及带 V-number 的脚本仅用于历史回放，不应被当作当前默认实验或 checkpoint
身份来源。

性能审计与 reader/DCT 的保留边界见：
`docs/performance/typed_reader_dct_performance_audit.md`。

已完成阶段的设计文档已归档到 `history_design/archive/`，实验脚本仍按版本保留在
`scripts/`，避免破坏历史复现实验。

Historical V53 replay entry point:

```bash
bash run_current_policy.sh
```

`run_current_policy.sh` dispatches to `scripts/current_v53_full.sh`.

## Historical V53 Replay Examples

The commands below reproduce archived V53 comparisons only. They are not the
active mainline and must not be used to start a Schema37 experiment.

### Experiment 1: V53-A + V53-B

Purpose: mainline vertical-depth run. This is the default setting of
`scripts/current_v53_full.sh`.

```bash
CUDA_VISIBLE_DEVICES=1 \
OUT_DIR=runs/v53ab_vertical_depth_b8 \
LATENT_CVAE_NOISY_GATE=1 \
LAYER_BOOST_RESIDUAL=1 \
LAYER_ZERO_BASE_DIAGNOSTIC=1 \
LATENT_CVAE_LAYER_SCAN=1 \
LATENT_CVAE_LAYER_SCAN_ALPHA=0.2 \
ADAPTIVE_CVAE_MONOTONIC_LAYER_ROUTE=1 \
ADAPTIVE_CVAE_LAYER_ROUTE_DISTANCE_SCALE=3.0 \
LATENT_CVAE_CANVAS_CROSS_ATTENTION=0 \
ADAPTIVE_CVAE_SERIAL_WRITERS=0 \
nohup bash run_current_policy.sh > v53ab_vertical_depth_b8.log 2>&1 &
```

### Experiment 2: V53-A + V53-B + V53-C

Purpose: test whether full-canvas cross-attention and serialized writers improve
trunk bandwidth without reintroducing shortcut behavior.

```bash
CUDA_VISIBLE_DEVICES=1 \
OUT_DIR=runs/v53abc_canvas_serial_b8 \
LATENT_CVAE_NOISY_GATE=1 \
LAYER_BOOST_RESIDUAL=1 \
LAYER_ZERO_BASE_DIAGNOSTIC=1 \
LATENT_CVAE_LAYER_SCAN=1 \
LATENT_CVAE_LAYER_SCAN_ALPHA=0.2 \
ADAPTIVE_CVAE_MONOTONIC_LAYER_ROUTE=1 \
ADAPTIVE_CVAE_LAYER_ROUTE_DISTANCE_SCALE=3.0 \
LATENT_CVAE_CANVAS_CROSS_ATTENTION=1 \
ADAPTIVE_CVAE_SERIAL_WRITERS=1 \
nohup bash run_current_policy.sh > v53abc_canvas_serial_b8.log 2>&1 &
```

If running both at the same time, put the second command on another free GPU.

## Historical V53 Key Log Fields

- `cxgate`, `xnorm`, `xratio`: direct `x_t` branch gate and branch strength.
- `cscan`, `clat`: depth-scan condition norm and lateral concat condition norm.
- `crmax`, `creff`, `cprmax`, `cpeff`: layer/progress route concentration.
- `czbase`: consequence zero-base shift; low values indicate action-feature parroting.
- `lboost`, `ldres`: residual magnitude learned by boosted layer contracts.
- `ctctrl`, `ctupd`, `ctsm`, `ctusm`, `ctue`, `ctpr`: trajectory projection/update diagnostics; smoothing/projection penalties are off by default.
- `pflow`, `rollout`, `delta`, `event`, `first8`, `tail`: main health metrics.
