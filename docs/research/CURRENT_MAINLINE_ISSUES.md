# ClearVLA Schema28 当前纯问题账本

更新：2026-08-31

本文件只保留完整 Schema28 行为运行之后仍未关闭的问题。当前执行图、动作
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

当前工作树已经实现无训练、无持久状态的 matched W/consequence/CT attribution，
并通过源码回归和 fresh CPU validation smoke。它尚未在上述 Schema28 checkpoint
上运行；本文件以下所有“责任尚未可分”结论因此保持不变，不得用 fresh-model
恒等测试替代 checkpoint 行为证据。

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

这些是结构与运行健康，不等于以下行为问题已经解决。

## V28-01 — 训练没有覆盖部署最终使用的 action-conditioned W 分布

源码事实：

```text
training
  coarse action -> W(coarse) -> one velocity call -> action loss

deployment / validation
  W(coarse) -> proposal ODE -> W(proposal) -> refined ODE -> final action
```

`training/engine.py::_forward_encoded` 只调用一次 `model.velocity`；loss 中的 W
future objective 也监督 encode 阶段缓存的 `W(coarse)`。训练没有让第二次策略调用
在自己预测动作重建的 W 上承担最终 action loss。这是已确认的 train/runtime
调用图错位。

它尚不是已确认的行为根因。现有日志没有 matched `W dynamic neutral`、
`ControlledTransition neutral` 与二者组合，因而不能证明只要增加一次训练侧
self-conditioning 就会改善远端或 gripper。

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

## V28-06 — W/P2 consequence 与 ControlledTransition 的责任尚未可分

真实源码路径是：

```text
W -> P2 typed effect -> protected consequence
                         |-> ControlledTransition context/action tokens
                         `-> Bottom protected read
```

W 已经到达 ControlledTransition，不存在缺失的 `W -> CT` 边。CT 也不是死路：
validation value RMS `1.4999`、spatial variation `0.6690`，训练尾部 raw gradient
约 `0.003`。再增加一条 W 直连会制造重复消费。

现有 `execution neutral/hard/full-capacity` 干预针对 execution controller，不是
W/CT matched responsibility 分解。还无法判断 CT 与 W consequence 是互补、重复，
还是分别负责不同动作频段。

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
