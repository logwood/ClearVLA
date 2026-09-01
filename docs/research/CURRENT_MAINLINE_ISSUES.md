# ClearVLA Schema29 当前纯问题账本

更新：2026-09-01

本文件只保留完整 Schema28 行为运行之后、进入 Schema29 后仍未关闭的问题。
在 Schema29 正式曲线产生前，Schema28 仍是行为基线，不是当前源码图。当前执行图、动作
ABI、运行频率与检查点边界见
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)，
下一版的实施顺序与放行门槛见
[`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md)。历史 donor
与已完成修复不在这里重演。

## 当前证据合同

正式证据来自
`new_logs/schema28_final_20260831_013140/{run_context.json,metrics.jsonl}`：

- source commit `097330a894d948d66c419f8af07325a5b0ff712e`，manifest Schema28；
- `object_intent_dynamics_323`，seed 0，batch 8，63/5 train/val episodes；
- action normalizer fingerprint `32a3a4d7f21f`；
- 8 个完整 epoch、1,144 个 train windows、无 traceback/non-finite；
- exact loss ledger；median `1.840 s/batch`，process peak `12.112 GiB`；
- validation sampling/P2/proposal coverage `8.94%`，execution matched coverage
  `4.47%`。低覆盖干预只能作局部归因，不能代替全验证行为结论。

Schema26 与 Schema28 的数据、split、normalizer、seed 和训练长度对齐，可作
近控制的版本方向比较；Schema27 只有 5 个完整 epoch，不能作为最终点对照。

正式 matched W/consequence/CT attribution 已在上述 Schema28 checkpoint 的全部
179 个 validation batch 上完成，16 个 diagnostic batch 通过所有 identity/coverage
门。W neutral 与 CT neutral 都有独立远端动作责任，二者不重复。随后 endpoint
estimator/full-proposal gate 以 `0.984706` 更新方向 cosine、`1.0` 有效覆盖通过，
放行 Schema29 detached self-conditioning。它们选择结构修改，不替代 Schema29
正式行为曲线。

Schema29 首轮 Pen/RDT 运行不是有效行为证据。真实 Pen batch 的 CUDA BF16 VJP
报告 `new_logs/schema29_real_batch_probe_a671640.json` 证明：cache0 单遍的
velocity/gripper/motion 参数梯度 L2 分别为
`3.1299548 / 0.01231698 / 0.05398325`，而实际 cache1 formal 路径三者全为
`0`；与此同时 physical-velocity 与 head-input activation VJP 在两边都保持
`7.79045e-4 / 3.24186e-6`。因此 finite loss、ledger、activation gradient 和
optimizer step 都掩盖了参数边被截断。旧实验已经停止，禁止续训。

## 已由本次运行关闭的边界

- Schema28 能稳定完成一次
  `proposal -> W(proposal) -> refined action`，不是空调用或 lineage 旁路；
- W 的 goal/coarse-hidden direct ingress 为零，显式
  `PhysicalActionCondition` 有非零 VJP；
- P2 semantic、geometry 和 geometry-address 的真实 consumed tensor 都有非零
  action gradient；
- typed-W floor 修正后没有再出现 typed normalization owner spike；
- capacity 的 FP32 路径、CandidateWorld identity、两遍 same-noise ODE、loss
  ledger、optimizer 和 checkpoint 合同均保持 finite/一致。
- matched attribution 已分开 W 与 CT：neutralize W dynamic 的 far/gripper action
  delta 为 `0.05829 / 0.13240`，neutralize CT 为 `0.01605 / 0.04142`，联合为
  `0.06527 / 0.15345`；二者都有独立责任，不新增 W->CT bridge。

这些是结构与运行健康，不等于以下行为问题已经解决。

## S29-00 — CUDA BF16 autocast 权重缓存截断 formal 参数梯度

活动训练在同一个外层 BF16 autocast 生命周期中执行：

```text
pass0 velocity under no_grad
  -> rebuild W
  -> pass1 formal velocity
  -> backward
```

PyTorch autocast 会缓存参数的 BF16 cast。pass0 是 dynamic path 第一次参数调用，
在 `no_grad` 下生成的 cached cast 没有参数边；pass1 随后复用它。forward 与
activation VJP 仍正常，只有 block/head 到原始参数的 VJP 变成零，所以原 smoke
只看 finite backward 不能发现该错误。

修复合同是不全局关闭 AMP cache，也不改变 dtype、loss 或数学图，而只在可能与
后续 attached 调用共享外层 autocast 的 parameterized no-grad scope 内嵌套
`cache_enabled=False`：训练 pass0、native candidate target probe、旧 sequential
learned-execution hard audit。formal path 仍使用正常 cache。源码候选已经通过
`166` 项相关 CPU 回归，并为两条内部路径证明 no-grad/cache-off 后 attached
block/head VJP 非零；CUDA 专项仍必须由远端通过。

关闭本条必须同时满足：

- v2 Pen B8 real-batch probe 中 pass0/condition 全 detached；
- pass0 cache off、cache1 formal cache on，formal dtype/forward/loss/RNG 不变；
- cache1 velocity/gripper/motion 与 Evidence-MMDiT block `0/1/2` 参数 VJP 非零，
  且相对 cache0 不出现 order-of-magnitude 强衰减；
- 新提交的 Pen B8 与 RDT-8 fresh smoke 都完成 backward/optimizer/checkpoint/
  deployment；
- 只用新空目录重启正式实验，旧 checkpoint exact resume 被 source digest 拒绝。

## S29-01 — train/runtime action-conditioned W 错位已结构修复，行为收益待验证

Schema28 的已确认错位是：

```text
training
  coarse action -> W(coarse) -> one velocity call -> action loss

deployment / validation
  W(coarse) -> proposal ODE -> W(proposal) -> refined ODE -> final action
```

Schema29 已改为一次 flow 采样、cache0 detached endpoint estimator、只重建 W、
cache1 正式 velocity；唯一 action loss 与 future loss 都消费 cache1。pass0 无
backward，forked RNG 令净随机流仍等于一个正式 dynamic pass。参数、optimizer、
loss 权重与部署两遍 ODE 不变，Schema28 exact resume 被拒绝。

这套调用与所有权结构在源码上成立，但首轮 CUDA 参数梯度并未闭合，见
S29-00。只有先通过新 VJP 门、再从 fresh state 得到 Pen 完整曲线，才能回答它
是否改善远端、gripper、final mismatch 与 spike；不得把旧 smoke finite、旧早期
aggregate 或修复后的首批 loss 当作行为关闭。

## V28-02 — outer refinement 是有效 correction，但不是自洽闭环

epoch 8 的 matched deployment 聚合为：

```text
proposal -> refined action delta RMS       0.02514
final interval vs W-condition mismatch     0.02933
final adjacent-delta mismatch              0.01514
W semantic change after proposal           0.06639
W transport change after proposal          0.00713
```

refinement 明确改变了 W 与动作，但最终动作又离开了其 W 的条件。interval
mismatch 与 action delta 使用不同聚合轴，不能作严格等式比较；它们处于同一量级
已经足以拒绝“fixed point/已闭环”的说法。只重算 `W(final action)` 而不再让策略
消费它，也不能把最终动作变成被该 world 条件化的结果。

## V28-03 — 远端和 gripper 仍是主要行为失败

Schema28 的最佳 aggregate 点在 epoch 6，final 有小幅回升：

```text
                         epoch 6       epoch 8
full physical RMSE       0.0751        0.07657
arm physical RMSE        0.0566        0.05677
gripper physical RMSE    0.1422        0.14733
tail / first             6.42          7.66
```

final horizon bands 为 `0.02502 / 0.05513 / 0.09743`，tail 为 `0.09016`；训练
flow 持续下降并没有消除 deployed tail gap。相对完整 Schema26 final，Schema28
aggregate、arm、gripper 和 tail/first 均方向性改善，但问题没有关闭。

decoded gripper 同时证明这不是单纯的尺度展示问题：precision `0.6006`、recall
`0.2749`、F1 `0.3771`，只预测 `621/1357` 个目标事件，event ratio `0.4576`。
远端 gripper band 为 `0.18233`；post-event `1-2 / 3-6 / 7+` 为
`0.34163 / 0.27129 / 0.17776`。下一版不能只改善 aggregate 或近端。

## V28-04 — W transport 相对 semantic 更欠拟合，但原因未归属

epoch 8 validation：

```text
W2 semantic / Teacher semantic       0.25078 / 0.36198 = 0.69x
W2 transport / Teacher transport     0.04072 / 0.09265 = 0.44x
```

transport 相对 Teacher 的缺口更大；W physical-action condition、dynamics 参数和
P2 consumer 均有非零梯度，所以不能称为断路。现有证据也不能区分：

- 训练只见 `W(coarse)` 导致条件分布不足；
- W objective/representation 本身不足；
- P2/consequence/Bottom 对 transport 的行为价值过滤；
- Teacher transport 的一部分在当前任务上并非动作必要信息。

不得据此直接调 transport loss、gain、quota 或 Huber 尺度。

## V28-05 — semantic 已承担明显动作责任，geometry 的角色很弱

deployment effect RMS 为 semantic `0.15870`、geometry `0.01820`，geometry
address correction 为 `0.01196`。matched P2 切片显示：

- neutralize geometry address 后 far arm/gripper action delta 只有
  `0.00050 / 0.00106`；
- zero geometry value 后 far-gripper action delta 约 `0.00729`；
- zero far semantic 后 far arm/gripper action delta 为 `0.02642 / 0.11407`，
  far-gripper RMSE 从 `0.12825` 恶化为 `0.20784`。

因此 semantic 的远端责任已经成立；geometry 当前行为效应很弱也成立。但现有
干预不能判定 geometry 应成为独立动作值、只应辅助 address，还是在这个任务上
本就信息量有限。直接放大 geometry 会改变 semantic/geometry 的竞争，依据不足。

## V28-07 — finite spike 显著减少，但 observation owner 仍会复发

完整 Schema28 共有 12 次 threshold `5.0` 的 finite preclip spike：

```text
observation flow delta_head                         4
observation target_dino_key                        4
bottom arm_abs output head                         3
observation raw-flow pyramid stem                  1
```

最大 global preclip 为 `24.63`（epoch 4 step 9118），owner 是
`target_dino_key`，owner L2 `13.49`。这比 Schema26 的 64 次、最大 `435.04`
明显缓和，但 observation 的两个旧 owner 仍复发；不能宣称数值问题已全部关闭。
若下一版相同 owner 再出现严重峰值，先对同 checkpoint/same batch 做 per-loss
VJP，不先加 clip、降 LR 或改 objective weight。

## V28-08 — execution capacity 基本全开，hard/soft 仍有差距

尾部 effective basis mass 约 `31.999/32`。matched full-capacity 对 primary 的
动作与 RMSE 几乎不变，而 hard execution RMSE `0.07737` 高于 matched primary
`0.07175`；neutral 更差至 `0.08750`。这说明 learned soft execution 有行为价值，
但 capacity 当前没有形成实质压缩。它不是下一版核心修复目标，也不能用硬化
capacity 来换取表面稀疏。

## 明确撤回和禁止的方案

- 撤回新增 `W -> ControlledTransition` bridge；已有 consequence 路径已到达 CT。
- 撤回“由 CT 生成 world1”；world producer 是 W，CT 只生成策略 transition。
- 不把第三次 `W(final action)` 重算冒充闭环；没有后续策略消费就没有条件闭合。
- 不直接放大 geometry、transport、gripper 或 interval-3。
- 不新增 hard event gate、entropy/route quota、额外 clip、人工梯度或 best-checkpoint
  选择器来掩盖完整八轮行为。

## 关闭规则

一个问题只有在 producer -> transform -> consumer -> loss 的正向审查与
consumer -> gradient owner -> optimizer -> producer 的反向审查均闭合，并且
matched intervention 改变了预定第一边界且有完整 coverage 后才能进入结构修改。
源码可解释但尚无行为证据时保留本条；日志幅度异常但责任未归属时不得直接改图。
混合精度路径还必须在真实 CUDA autocast 生命周期内验证原始参数 VJP；非零
activation gradient、finite global norm、optimizer.step 被调用或 CPU backward
均不能替代该门。
