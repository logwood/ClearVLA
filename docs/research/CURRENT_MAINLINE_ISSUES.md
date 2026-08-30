# ClearVLA 当前纯问题账本

更新：2026-08-31

本文件只保留 Schema28 尚未由行为证据关闭的问题。已采用的图、动作 ABI、
运行频率和检查点身份见
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)。
实现来源与回滚边界保留在相应 auxiliary 审计中，不在这里复制历史方案。

## 当前证据边界

- 行为参考仍是 V120 `long`，提交
  `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`。
- 最新完整度较高的行为样本仍是 Schema26：三个完整 validation 加
  epoch-4 batch 360；它不能证明 Schema27 或 Schema28 的行为。
- Schema27 只完成了本地 source/CPU BF16 验证，没有 CUDA behavior run。
- 当前源码为 Schema28。它实现的是一次有界
  `proposal -> W(proposal) -> refined action`，不是 fixed point、跨控制周期
  belief loop 或机器人环境闭环。

## P0：一次 outer refinement 是否真正产生有用而非陈旧的后果修正

Schema28 已经从结构上关闭以下旧缺口：W 只读显式物理动作条件，goal/S/coarse
hidden 不再进入 W；CandidateWorld 与动作条件原子绑定；validation/deployment
使用同一初始噪声跑 proposal 和 refined 两遍 ODE。

仍未知的是行为必要性。第二遍最终动作可以再次偏离用于 W 重算的 proposal，
因此 `final_world_action_*_mismatch_rms` 是必须观察的 residual，而不是应被强制
归零的 loss。

### 关闭条件

- CUDA smoke 中 refinement count、tag identity、pre/post action、semantic、
  transport 和 final mismatch 全部存在且 finite；
- 后果敏感样本上，正确 CandidateWorld 优于 wrong-action/wrong-world 对照；
- refined action 相对 proposal 的变化不只是公共偏移，并且不能靠 aggregate
  RMSE 掩盖 arm、gripper 或 tail 退化；
- 若 final mismatch 持续与 proposal-to-refined change 同量级，本版只能判为
  一次 correction，不能声称自洽闭环。

## P0：W 与 ControlledTransition 仍是两个未统一的动力学责任

W 现在拥有 `ObjectWorldBelief + PhysicalActionCondition -> CandidateWorld`，但
`ControlledTransition` 仍在每个 ODE 节点读取 noisy action、consequence 和 P1
residual，并直接进入 Bottom。Schema28 没有把它映射到 W 的对象 future，也没有
增加一致性目标。

### 关闭条件

- matched W/P2/ControlledTransition 干预明确各自的动作责任；
- 仅保留 ControlledTransition 不能在后果敏感切片上解释全部行为；
- 后续实现必须先给出 object/time 轴上的同义映射，不能靠额外 gain、loss quota
  或强行删除 P1 来制造依赖。

## P1：gripper transition/persistence 修正不能以 arm 或 tail 为代价

Schema26 的 gripper physical RMSE 到 epoch 3 改善为 `.1593`，但 arm 为 `.0656`，
落后于 R2 epoch 3 的 `.0621`；tail/first 仍高。Schema28 将事件行 transition 与
事件间 persistence 分离，并在每次事件重新锚定，修复了旧累计 delta 可跨事件和
前事件泄漏的问题，但尚无 CUDA 行为证据。

### 关闭条件

- 同时报告 absolute/delta branch、transition/persistence loss、post-event
  `1-2 / 3-6 / 7+`、decoded event ratio/F1；
- gripper 改善不能伴随 arm、first、tail 或远端 band 明显回退；
- 不用 event gate、类别头或新权重掩盖连续物理场问题。

## P1：capacity 的 BF16 exact-one 死区需要运行时证据

Schema28 保留原控制器与 schedule，只让 capacity head 和插值在 FP32 到达
contraction。源码和单测能证明 near-one 差异不被 BF16 直接舍入，但不能证明
训练后 controller/capacity 对任务有用。

### 关闭条件

- CUDA 中 capacity 不长期精确等于 1，non-expansive violation 保持零；
- capacity/operator-basis 梯度在 execution warmup 后 finite 且非长期零；
- matched `full_capacity` 和 `three_basis_reduction` action delta 有覆盖，解释时
  与 execution policy、有效 basis mass 和任务误差一起看。

## P1：spike 来源、geometry 与远端 W 仍未由 Schema28 关闭

Schema26 的 64 次 finite spike 显著多于 R2 的 15 次，最大 global preclip 为
`435.04`。Schema27 修复了 typed normalization 的一个确定性放大边界，但没有
CUDA 结果。Schema28 又改变了 W 输入责任，因此旧日志只能给风险先验。

geometry value/action effect 与 W2 transport 是否恢复，也必须在新的显式动作条件
下重测；不能把 Schema26 的弱幅度直接外推到 Schema28。

### 关闭条件

- 完整记录 spike owner、parameter、L2/max-abs、batch offset 与 global preclip；
- W physical-action VJP、G typed fact VJP、W2/Teacher transport、P2 semantic/
  geometry effect 和 matched intervention 同时存在；
- 若 spike 或 geometry 仍异常，再用同 checkpoint/same batch 的 per-loss VJP
  判因，不先调 gain、clip、LR 或 objective weight。

## P2：null/confidence、时间因果、persistent belief 和新观测反馈尚未实现

Schema28 没有引入 P2 对象/interval null、future confidence、可重建 48-step
动作时间结构、跨控制周期 track identity、birth/death/occlusion、执行前缀或
Observation(t+1) belief update。当前 K=4 仍是单次 observation 内的全局对象槽。

这些是后续主线问题，但不应被伪造为本轮已完成字段，也不应在没有真实 consumer
时先加入 checkpoint 状态或日志占位。

## 当前放行门槛

1. 本地完整 mainline/auditor suite、py_compile、ruff、diff check、生产维度 CPU
   BF16 单 batch 与双遍部署均通过；
2. fresh CUDA BF16 smoke 验证双遍 ODE、一次 W 重算、same-noise、finite residual、
   capacity 与显存；
3. smoke 审计通过后才启动 fresh batch-eight 正式训练；
4. batch-2200 做第一次同数据/normalizer/seed 合同的健康比较；
5. 完整八轮同时比较 V120、R2、Schema26、Schema27，不挑 best epoch；
6. 任何“闭环完成”结论都必须明确限定为一次 action-world correction，直到
   ControlledTransition、persistent belief 和新观测反馈真正关闭。
