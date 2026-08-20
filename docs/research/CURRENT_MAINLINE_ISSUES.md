# ClearVLA 当前纯问题账本

更新：2026-08-20

本文件只保留当前仍未解决的问题。已在 Schema25 源码中关闭的 S 公共化、跨类型 learned-null 竞争以及 CoarseAction/W raw-typed 旁路已经移除；实现决定见
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)，实验验收边界见
[`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md)。

## 证据边界

- 行为参考：V120 `long`，提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`，完整快照 `.audit/v120_exact_source_0b92d359/`。
- 上一完整实验：Schema24 `schema24_fidelity_fix_b8.log`，八个 epoch 全部结束。
- 当前源码：Schema25，能力名 `object_intent_dynamics_323`；122 个本地 mainline 测试通过（连同日志审计共 153 项），但 CUDA smoke 与 fresh 八轮尚未运行。
- 因而本文可以确认源码依赖是否修正，不能预先声称动作性能已经改善。

## P0（独立暂缓）：最终 G3 anchor 事实轴在 transition source 前被抹掉

### 已确认源码事实

- G1→G2→G3 已形成 `[B,4,C,8,8,H]` anchor-aware grounding rollout。
- 当前 transition source 没有直接消费该四 anchor chart；它从 public chart 构造一份公共 128-row 内容，再加四组 identity label 得到 512 rows。
- 因此 512-row shape 正确，但四组内容不是四个真实 G3 anchor facts。旧 V120 则把最终 grounding rollout 直接交给 controlled dynamics。

### 为什么暂缓

该缺陷与本轮 S 所有权修复相互独立。同时修改会使 fresh 动作变化无法归因。Schema25 的第一轮受控实验只验证 S；不得因为它邻接 S 就顺手重写 transition 或 bottom。

### 关闭条件

- transition flatten 前张量与最终 G3 rollout bit-exact；不能由公共 chart 加 identity label 近似。
- G3 anchor permutation 必须在 transition source 中等变。
- 不改变 transition 的 512-row ABI、动态更新频率、CVAE/workspace/Evidence MMDiT 或 execution。
- 单独 fresh smoke 与完整对照通过后，才从本账本删除。

## P1（行为未归因）：epoch 7/8 的 gripper 与中远程回退

Schema24 的 physical RMSE 在 epoch 6–8 为 `0.08008 / 0.08193 / 0.08218`；回退主要来自 gripper 与 5–24 步，arm 基本稳定。V120 自身也从 epoch 7 的约 `0.0793` 回升到 epoch 8 的约 `0.0814`，所以晚期反弹不是已被证明的单一 S 故障。

同期 Schema24 的 S typed innovation 从 epoch 6 的 `0.00350` 回到 `0.01004 / 0.00860`，W interval 指标没有突变；P2 null mass 从 `0.1270` 降至约 `0.0946`，consequence/effect RMS 增长约 `10%`，但没有冻结干预证明该同步漂移是因果根源。

### 当前处理

- 不在 Schema25 源码中调 P2/P3 gain、null、route mass、loss 或 hard gate。
- fresh 八轮必须同时比较最佳点与最终点、first/tail、1–4/5–12/13–24、arm/gripper、event/motion。
- 若 S 边界健康但晚期回退仍在，继续归类为独立泛化/数据可识别性问题，不再用接线补丁解释。
- 若 P2/consequence effect 继续增大而 W target-normalized error 与动作验证不改善，视为拒绝信号，不视为“使用率提高”。

## 当前放行阻塞

Schema25 只有完成以下外部实验后才能视为主线候选：

1. fresh CUDA BF16 smoke；
2. batch 8 进程显存不超过 22 GiB，并记录吞吐；
3. 完整八个 epoch，与 Schema24 和 V120 全指标对照；
4. per-type S/W 边界在真实日志中均存在且有限，不能只剩一个合并 RMS；
5. 不能同时恶化最佳 physical RMSE 与最终 physical/gripper/中远程指标。

在上述结果返回前，不新增 S/W block、阶段标签、scalar progress、使用率配额、entropy/diversity loss 或人工梯度。
