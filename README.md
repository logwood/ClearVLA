# ClearVLA Current Policy Run

## Current Object Intent–Dynamics 3-2-3 Mainline

The current experiment candidate is the `object_intent_dynamics_323`
capability, logged as V121. The numeric label is bookkeeping; source selection
and checkpoint identity use the capability manifest. The top graph has K=4
global grounded objects plus an explicit null, a stateless factorized online
intent organizer plus a training-only future recognizer, W1/W2 object dynamics
over `4-8 / 8-16 / 16-32 / 32-48`, one P1 high-resolution factual read,
bounded zero-preserving semantic/geometry P2 routing through an
`ObjectFactualDock`, and three real P3 innovations: precision, temporal and
zero-centred observable state change. The Evidence MMDiT/CVAE/workspace bottom
remains intact behind exactly one protected-consequence ingress.

Batch-eight memory smoke (diagnostic synchronization is enabled, so do not use
its timing as throughput):

```bash
CUDA_VISIBLE_DEVICES=0 \
OBJECT_323_BATCH_SIZE=8 \
SMOKE_TRAIN_BATCHES=2 \
OUT_DIR=runs/v121_object_323_typed_dock_b8_memory_smoke \
nohup bash scripts/current_object_intent_dynamics_323_smoke.sh \
  > v121_object_323_typed_dock_b8_memory_smoke.log 2>&1 &
```

Long run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OBJECT_323_BATCH_SIZE=8 \
OUT_DIR=runs/v121_object_intent_dynamics_323_typed_dock_b8 \
nohup bash scripts/current_object_intent_dynamics_323.sh \
  > v121_object_323_typed_dock_b8.log 2>&1 &
```

The raw HDF5 default remains
`/data/liang.zhang/dataset/grab_pen_single/grab_pen_single`. The current
launchers use `/data/senwang/data` only for decoded/DINO caches and
`/data/senwang/checkpoint` for T5/model weights. `DATA_ROOT`, `CACHE_DIR`,
`DINO_CACHE_DIR` and `T5_CONDITION_PATH` remain individually overridable.
Top resume is intentionally rejected because the object graph is fresh-only.

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

当前 policy 主线已经迁移到 `clearvla/policy/`；训练与烟测入口就是上一节的两个
capability 脚本。V53、V87、V118 和 V119 脚本只保留作历史回放，
不应被当作当前默认实验。

性能审计与 reader/DCT 的保留边界见：
`docs/performance/typed_reader_dct_performance_audit.md`。

已完成阶段的设计文档已归档到 `history_design/archive/`，实验脚本仍按版本保留在
`scripts/`，避免破坏历史复现实验。

Historical V53 replay entry point:

```bash
bash run_current_policy.sh
```

`run_current_policy.sh` dispatches to `scripts/current_v53_full.sh`.

## Recommended Experiments

Run these as controlled comparisons. V53-A and V53-B are the default mainline;
the second experiment adds V53-C only.

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

## Key Log Fields

- `cxgate`, `xnorm`, `xratio`: direct `x_t` branch gate and branch strength.
- `cscan`, `clat`: depth-scan condition norm and lateral concat condition norm.
- `crmax`, `creff`, `cprmax`, `cpeff`: layer/progress route concentration.
- `czbase`: consequence zero-base shift; low values indicate action-feature parroting.
- `lboost`, `ldres`: residual magnitude learned by boosted layer contracts.
- `ctctrl`, `ctupd`, `ctsm`, `ctusm`, `ctue`, `ctpr`: trajectory projection/update diagnostics; smoothing/projection penalties are off by default.
- `pflow`, `rollout`, `delta`, `event`, `first8`, `tail`: main health metrics.
