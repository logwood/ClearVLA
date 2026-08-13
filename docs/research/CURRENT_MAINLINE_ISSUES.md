# ClearVLA 当前纯问题账本

更新：2026-08-13

本文件只记录 Schema24 源码完成后仍未关闭的问题。已经修复的
G1→G2→G3、V120 P1、grounder 目标、对象级 W/P2 几何、optimizer decay、
局部/全局裁剪和梯度日志问题已删除；它们的历史依据保留在审计账本与 Git
差异中，不在这里反复占用注意力。

## P0：尚未完成生产环境行为验收

### 当前事实

本地 CPU/FP32/BF16 边界测试只能证明接口、autograd 和数值契约闭合，不能证明
24 GiB GPU 上的真实显存、吞吐和八轮泛化已经恢复。Schema24 因此仍是
“源码完成、实验待验收”，不是已经接受的实验基线。

### 关闭条件

- fresh BF16 smoke 和五步部署通过；
- batch 8 总进程峰值不超过 22 GiB；
- 对齐 batch 2200，与 V120 比较 G3 parent L1、object pair cosine、
  P1 spatial variation、P2 null mass 和 P2 effect RMS；每项至少关闭
  Schema23→V120 差距的 50%，且没有指标继续远离 V120；审计命令必须同时
  提供 `--recovery-baseline` 与 `--recovery-parent`，不能拿 V120 第八轮 tail
  和 Schema24 第一轮比较；
- 跑满八个 epoch，比较全部 train/validation、first/tail、四 horizon、
  arm/gripper、event/motion、G/S/W/P、raw/postlocal/postglobal 梯度；
- 最终点和八轮均值均通过 recovery gate，不能用 best checkpoint 掩盖后期反弹。

## P1：参数、显存和吞吐差异仍需由真实启动清单解释

源码已经在 run context 中写入逐模块参数量，不再硬编码总参数。尚需生产 smoke
记录并解释：

- progressive G1-G3、exact V120 P1 与删除旧 host/K-object P1/额外 grounder
  heads 各自带来的参数差异；
- batch 1 与 batch 8 的模型、activation、reserved 与 context 峰值；
- Teacher 每训练 batch 一次、部署零次；
- 静态 observation/G/S/W/P1 一次，动态 P1/P2/P3/transition/bottom 六次
  （五个更新节点加一个 endpoint head）。

若时间或显存不满足边界，先用调用次数和模块清单定位；不得先缩弱 P1、删除
bottom、减少 N=49 或降低高分辨率读取来制造“优化”。

## P1：结构恢复后 S/W/P3 的可识别性尚未由长跑判定

旧日志中的 S 公共化、W 比 Teacher 更公共、effect 使用弱、P3 temporal/history
偏置可能来自此前错误的 G/P1/geometry/训练生命周期，也可能在正确接线后仍然
存在数据可识别性问题。源码审查目前没有依据继续改 S/W/P3。

只有在以下边界全部正确且 Schema24 长跑仍复现问题时，才建立下一轮结构任务：

- G1/G2/G3、P1 N=49/microgrid、对象几何和 support/selector 探针均通过；
- W zero/shuffle 先显著改变 P2/effect/consequence 边界；
- action 端置信区间仍跨零，或 W zero 仍改善动作；
- S/W 的区间变化在完整轨迹与完整 epoch 上仍公共化。

届时应归类为“监督/可识别性”而不是继续补接线，不允许用 gain、quota、硬门控、
entropy loss 或人工梯度制造使用率。

## P2：future_address 是无消费者的诊断债务

`FutureObjectDynamics.future_address [B,4,K,C,8,8]` 只用于诊断与 Teacher
可视化；在线 P2 使用对象级 coordinate/transport/validity，不消费该张量。
它当前不改变 action，也不是本轮回归原因。

后续若删除，必须先确认日志、探针和 checkpoint schema 不再依赖；不得把它重新
接入 P2 以制造相机轴或空间使用。

## P2：P1 learned null 明确延期

当前 exact V120 P1 没有 learned null，protected factual base 不参与可选竞争。
若未来引入，只允许：

- protected base 位于竞争外；
- null value 代数精确为零；
- null 只抑制 optional detail innovation；
- 先验只来自当前可观测证据；
- 不读取 noisy action、future Teacher 或直接 policy carrier。

在出现明确的 detail 误读证据前，不实施该机制。

## P2：future proposal 的主路 no-op 保持为显式设计

HistoryActionProposal 仍有监督，但其未来预测 token 不进入 G/S/W/P/transition/
bottom；真实可观测 executed-action history 通过共享 V120 seed 进入主路。当前
不删除该辅助项，也不把 proposal 接回主路。只有完整消融证明辅助 loss 无益且
产生显著开销时，才单独处理。

## 失败时的归类规则

- 若 smoke 在 forward/shape/dtype 失败：源码契约问题，停止长跑并修复；
- 若出现 non-finite：以 `gradient_failure` 首个参数记录归因，不先加 clip；
- 若显存/速度失败：按静态/动态调用次数与模块清单归因；
- 若结构边界通过但八轮性能失败：数据/监督/可识别性问题；
- 不把“梯度存在”“tensor 非空”或“辅助 loss 下降”当作主路有效证据。
