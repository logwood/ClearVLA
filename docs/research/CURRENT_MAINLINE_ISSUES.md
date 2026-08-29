# ClearVLA 当前纯问题账本

更新：2026-08-29

本文件只保留当前仍未由行为证据关闭的问题。已采用的图和数值边界见
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)；
来源、反向路径和实现假设见
[`auxiliary/SCHEMA25_R2_SOURCE_DERIVED_IMPROVEMENT_PLAN.md`](auxiliary/SCHEMA25_R2_SOURCE_DERIVED_IMPROVEMENT_PLAN.md)。
B-spline 与 MIP 仍是互相独立的 auxiliary 候选，不属于本轮问题修复。

## 当前证据边界

- 行为参考：V120 `long`，提交
  `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`。
- 当前行为样本：Schema26，提交
  `33708a4ef3dfbf15ffd0ae9483527fa6c30d0ea9`；日志包含 447 个训练窗口、
  64 个 finite spike、三个完整 epoch 和 epoch-4 batch 360，没有 epoch-4
  validation。
- 当前源码：Schema27。它只修复 typed W 的归一化边界；public/generic W、
  参数、state key、optimizer、RNG、loss 与运行频率不变。focused source/
  identity/gradient tests 210/210 已通过；production-dimension CPU BF16 单
  batch 与 retained five-step deployment 已通过，CUDA behavior 尚未运行。
- 因而 Schema26 日志可以判定问题与方向，不能证明 Schema27 已改善任务性能
  或 spike。

## P0：finite gradient spike 明显增多，但尚未唯一判因

Schema26 有 64 次 spike，最大 global preclip `435.04`；R2 为 15 次、最大
`16.10`。owner 分布为 observation `target_dino_key` 23 次、observation flow
`delta_head` 13 次、gripper delta 13 次、arm head 9 次，其余 6 次。

源码确认 typed W 的普通 LayerNorm 会把低至 `4.55e-4` 的 interval 输入扩成
约 `.3` 量级状态，并允许接近 `316` 的局部归一化增益。这是一个独立成立的
结构错误，Schema27 已把 typed-only gain 限为 `4`；但 R2 也拥有旧算子且 spike
更少，所以不能宣称它解释全部 observation spike。

### 关闭条件

- Schema27 fresh smoke 与正式训练保持 finite；
- normalization denominator 不低于 `.25`、logged gain 不高于 `4`；
- observation spike 次数/幅度显著回到 R2/Schema26 可接受区间，或通过相同
  checkpoint、相同 batch 的 per-loss VJP 找到另一条明确来源；
- 不以新增 clip、loss quota 或降低学习率掩盖来源。

## P1：Schema26 的 gripper 改善伴随 arm 落后

Schema26 epoch 1/2/3 的 full/arm/gripper physical RMSE 为：

```text
.0921 / .0729 / .1658
.0919 / .0702 / .1721
.0856 / .0656 / .1593
```

相对 R2 epoch 3，gripper 从 `.1759` 改善到 `.1593`，arm 则从 `.0621` 退到
`.0656`。tail/first 从 `2.39` 增到 `4.49`，仍有明显长程误差。当前日志不含
epoch 4 validation，不能判断这是暂时平台还是后续反弹。

### 当前处理

- 不回滚连续 gripper trajectory，也不新增 event gate；
- 下一次对照必须同时看 arm、三个 horizon band、两条 gripper branch、
  post-event bins 与 decoded-event ratio；
- 只有完整八轮才能判断 Schema27 是否保留 gripper 收益而恢复 arm。

## P2：geometry 仍弱，需先经过新的 W 数值边界再判

Schema26 epoch 1/2/3 的 W2 transport/Teacher 约为
`.0196/.0792`、`.0187/.0761`、`.0217/.0822`；P2 geometry/semantic effect 约为
`.00985/.0829`、`.00897/.1077`、`.00780/.1160`。geometry address correction
确实到达 action，但作用仍接近零。

当前源码审计没有发现 P2 metric floor、covariance `I+covariance`、transport
value projection 或 spatial-to-terminal 中的第二个 inverse-small normalization。
因此先修 W typed amplitude semantics。若 Schema27 中 W transport 与 geometry
effect 仍弱，才重新打开 transport head/objective 责任边界；在此之前不加 gain、
quota、hard interval mask 或 geometry loss 权重。

## 当前放行阻塞

Schema27 只有完成以下项目后才能视为主线候选：

1. fresh CUDA BF16 smoke，batch-eight process peak 不超过 22 GiB；
2. 完整八个 epoch，与 R2、Schema26 和 V120 比较，而不是只选 best epoch；
3. typed norm denominator/gain/RMS ratio 与 W-only ingress VJP 全程 finite；
4. spike、arm、gripper、近/中/远程没有通过相互牺牲制造 aggregate 改善；
5. geometry 是否继续处理，只由 Schema27 的 W/Teacher、P2 causal effect 和任务
   指标共同决定。
