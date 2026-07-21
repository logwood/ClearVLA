# ClearVLA Current Policy Run

## Current Layout

当前 policy 主线已经迁移到 `clearvla/policy/`。当前 DCT/typed-reader 实验入口是
`scripts/current_v87_spectral_controller.sh`；根目录下的 V53 命令保留为历史回放参考，
不应被当作当前默认实验。

性能审计与 reader/DCT 的保留边界见：
`docs/performance/typed_reader_dct_performance_audit.md`。

已完成阶段的设计文档已归档到 `history_design/archive/`，实验脚本仍按版本保留在
`scripts/`，避免破坏历史复现实验。

Current entry point:

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
