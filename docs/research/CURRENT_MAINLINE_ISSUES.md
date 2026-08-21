# ClearVLA 当前主线纯问题账本

更新：2026-08-21

当前源码身份：Schema29 `object_intent_dynamics_323`。行为比较锚点是
V120 `long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 和本地完整
快照 `.audit/v120_exact_source_0b92d359/`。V120 是行为锚点，不是正确性公理。

本文件只记录当前源码仍未解决的问题。已实现的结构决定写在
`00_CURRENT_ARCHITECTURE_CONTRACT.md`；旧 schema 的已修故障不在这里保留历史副本。

## 记账规则

- 源码可直接证明的旁路、轴丢失、错误默认值或梯度生命周期才称为确定性故障。
- 曲线相关性不单独证明因果；没有冻结干预时明确写“动作影响未知”。
- 张量存在、梯度非零、loss 下降都不等于策略正在使用该边界。
- 不用 gain、quota、hard gate、熵/多样性目标、额外外部 loss 或人工梯度掩盖问题。
- Schema29 必须 fresh run；Schema28 及更旧 checkpoint 不允许 exact resume。

## O-01：S public future-state 监督与 W public working coordinate 仍不是同一物理边界

**类型：残余所有权风险；typed semantic/status/transport 分叉已在 Schema28 关闭。置信度：中高。**

Schema28 的 typed 监督直接解码 W 实际使用的 pre-W `FutureObjectDynamics` 边界，旧的
独立 semantic/status/transport heads 已删除。但 public state 仍是：

```text
state_head(S interval_condition_innovation) -> future state summaries
S interval_condition_innovation             -> W public working block
```

`state_head` 只约束低维 row-space，W 仍可在其零空间内形成公共 carrier。该目标还是四区间
绝对 future-state 均值，单任务下可能比对象变化更容易优化。它不构成 Teacher 泄漏，也没有
自由 W residual 进入 P；问题是“public loss 降低”仍不能证明 W public coordinate 有区间/对象
辨识力。

关闭条件：按 goal/history/G intervention 记录 S condition innovation、public state prediction、
W1/W2 public state与最终 field 的链式 JVP；完整验证集中 W prediction/Teacher interval variation
和 adjacent cosine 必须改善，不能只看 public loss。

## O-02：P1 动态 action self-write 可能压过缓存的高分辨率事实

**类型：V120 祖传结构风险；是否伤害 action 尚未由冻结干预证明。置信度：中高。**

当前仍保留准确的 V120 代数：

```text
canvas        = action_query + protected_detail
dynamic_delta = P1_policy_block(canvas) - canvas
completed_P1  = protected_detail + dynamic_delta
```

Schema27 后期 `protected_detail≈0.0338`、`dynamic_delta≈0.244--0.256`，动态写入约为静态
detail 的 `7x`。Schema28 没有重写 P1；`FactualPrecisionDock` 也不是替代 reader，它缓存的
就是原 24-query/N=49/3×3 reader write。风险在于动态 delta 是否在方向上淹没它。

关闭条件：同年龄对比 V120 的 protected/dynamic/self/FFN RMS，并在冻结 checkpoint 上做
detail zero/shuffle 与 action-query shuffle，先观察 completed P1，再观察 action。没有链式证据
前不拆 P1、不缩减高分辨率读取。

## O-03：W successor 与 semantic-delta 仍是同一误差的重复目标

**类型：V120 祖传确定性目标冗余；对性能的净影响未知。置信度：高。**

两侧都满足 `successor=current_reference+semantic_delta`，且 current reference 相同。因此
successor error 与 delta raw error 代数等价，只是 delta 另加 scale-normalized/direction 项。
future loss 内它们分别占 `0.30/0.25`，会重复强调 semantic，相比 transport/status 的有效
压力更强。S typed direct loss 的 semantic/status/transport 也未按 target scale 对齐。

关闭条件：先做分字段 action JVP/干预，证明重复 semantic 压力确实挤压 geometry/status；若
成立，只保留一个 delta 误差或把 successor 改成独立稳定内容目标，且 future 外部总权重不增加。

## O-04：Teacher 的低置信、广空间 posterior 仍可成为全权重 future content target

**类型：V120 祖传目标质量缺口；当前日志证明活跃。置信度：高。**

Teacher reliability 不参与正式 loss mask，这是防止 reliability shortcut 的正确边界；但
`P(null)` 低而空间 entropy 高时，successor 会成为大范围 future DINO 软平均并获得与尖锐匹配
相同的 content loss 权重。Schema27 epoch 6 曾出现 null `≈0.055`、association confidence
`≈0.298`、reliability `≈0.290`，属于这一状态。

关闭条件：记录 effective support、top-k mass、best-minus-background、匹配半径、跨相机质量及
其与 target interval/object variation 的关系。只有能区分“有效多假设”和“无辨识空间平均”后，
才允许把后者退化为 identity + high uncertainty；不得重新用 reliability 标量静默缩 loss。

## O-05：P2 的候选集合与 explicit null 先验仍未被实验识别

**类型：模型选择风险，不是当前确定性接线故障。置信度：中。**

Schema28 已删除 Schema27 的固定 `-log(16)` 半空先验，并把 semantic/geometry/status 分成三个
独立 posterior；status 使用 current support，geometry 可提供正匹配证据。现在每类仍是 16 个
合法 interval×K 候选与一个 null 直接竞争，等分 logits 会天然偏向候选集合。由于所有 effect
value 都严格零中心，这不会在 neutral field 下创造 action value，但可能使 null mass难解释。

关闭条件：用 per-type source/null 与 effect zero/shuffle 查看是证据还是候选数量决定路由。没有
动作与 consequence 证据前，不再增加集合校正常数、learned gate 或质量约束。

## O-06：bottom 的 neutral generic trajectory 经带偏置适配器后仍可能成为 learned constant

**类型：V120 祖传确定性常量 source；实际采用程度未知。置信度：高。**

`owned_trajectory_memory` 为零，但 EvidenceViewAdapter 的 affine LayerNorm/Linear 可把零映射为
非零常量，并以 geom identity 进入 evidence bank。它可能只是 V120 的合法数据集先验，也可能
吸收应由 G/P1 提供的 attention；当前没有 source-level JVP 不能判断。Schema28 不修改 bottom。

关闭条件：投影 trajectory/rollout attention 与 JVP；若该 source 只承担 null，value 必须精确零
且与 factual source 分离；若只是 ABI 空行，应结构性排除。不得用负 bias 或 quota 强迫少读。

## O-07：learned flow 的几何质量在 Schema25→27 同年龄比较中持续退化

**类型：已观测训练问题；与顶层所有权问题可能并存但不能生造单一因果。置信度：高。**

既有同 iter 日志显示 learned flow 相对 native 的差距扩大、warp/cycle 与 confidence 没有同步
改善。Schema28 没有改 flow 模块、几何 loss 或其权重，只修复 G camera evidence 与 W/P2
geometry 消费。因此新版本若仅下游使用改善而 flow 本身仍退化，本项仍成立。

关闭条件：同 iter 比较 native/learned、warp/cycle/smooth/uncertainty、flow magnitude/confidence、
G geometry variation 和 P2 geometry posterior；必要时再独立审查 flow 优化，不把下游结构修复
误报为 flow 已恢复。

## O-08：后期 tail/gripper 回弹仍未归因

**类型：完整训练泛化问题。置信度：高；结构独占因果未知。**

V120、Schema24--27 都出现过早期下降后中远程或 gripper 回弹。S/W 公共化、P1 self-write、
Teacher 质量、数据中 gripper event 稀疏都可能贡献，但现有证据不能把它归给单一模块。

关闭条件：Schema28 必须完成八轮；同时看 train/val action、first/tail、四 horizon bands、arm、
gripper、event/motion 和 condition-keep 分层。不能用 best checkpoint 或前 2200 batch 代替全程。

## O-09：当前因果日志仍不足以把“边界改善”升级为“动作收益”

**类型：可观测性债务。置信度：高。**

Schema28 已增加 conditional-K、camera evidence、G/S/W public-vs-K innovation、
S common/differential denominator、W typed state、per-type P2 和 P3 operand 指标；
frame-progress audit 只读取真实 condition innovation。结构测试覆盖 exact zero、permutation 和
forbidden input。但普通
nohup 仍不能替代冻结 checkpoint 的链式干预，尤其是 protected P1 vs precision、三类 W field、
四个 P3 source 和 bottom neutral trajectory。

关闭条件：smoke 通过后使用冻结 checkpoint 分层 zero/shuffle，并按
`boundary -> P2/consequence -> bottom source -> action` 报告效应与置信区间。只有边界和最终 action
都离开零，才声称策略实际使用该信息。

## Schema28 放行检查

- fresh BF16 smoke；旧 schema exact resume 必须拒绝；Teacher 部署调用为零；
- batch 8 总进程显存不超过 22 GiB，并记录吞吐；
- 对齐 batch 2200 比较 V120/Schema27：conditional-K entropy、camera evidence、
  G/S/W public-vs-K innovation、S condition/typed、W per-type variation、
  P2 per-type null/interval mass、P3 precision/effect/temporal；
- 完成八个 epoch并检查 O-07/O-08；
- 若接线边界健康但 action 无收益，归类为数据/可识别性问题，不继续叠加结构补丁。
