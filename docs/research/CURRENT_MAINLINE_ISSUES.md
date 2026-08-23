# ClearVLA 当前主线纯问题账本

更新：2026-08-23

当前源码身份：Schema33 `object_intent_dynamics_323`。行为比较锚点仍是 V120
`long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 与本地完整快照
`.audit/v120_exact_source_0b92d359/`。V120 是行为锚点，不是正确性公理。

本文件只记录当前源码仍未解决的问题。已经落地的 canonical G、S 单一状态监督、
W-owned common/residual、W2 residual-only bridge、共享时间/typed-object P2、
camera-mixture P2、Teacher partial assignment 与目标代数
属于当前架构，不在这里保留旧故障副本。

## 记账规则

- 日志证明问题是否活跃及量级；源码证明数据流、旁路和目标代数。
- 源码可直接证明的旁路、轴丢失、错误默认值或生命周期错误，才称确定性故障。
- 曲线相关性不单独证明因果；没有冻结干预时明确写“动作影响未知”。
- 张量存在、梯度非零、loss 下降都不等于策略正在使用该边界。
- 不用 gain、quota、hard gate、熵/多样性目标、额外外部 loss 或人工梯度掩盖问题。
- Schema33 必须 fresh run；Schema32 及更旧 checkpoint 不允许 exact resume。

## O-01：global-K 绑定后的有界校正器几乎没有实际 assignment 权限

**类型：活跃的功能近空转风险；不是整个 G1–G3 坍塌。置信度：高；动作影响未知。**

源码锚点：`clearvla/mainline/model/grounding.py::DenseObjectGrounder.forward`。

当前 grounder 先得到 parent K posterior，再从 `g3_residual` 去掉公共分量，只校正
conditional-K：

```text
raw residual -> subtract weighted common -> add to log(parent K conditional) -> softmax
```

该代数正确保护 object-vs-null mass，但 Schema30 日志中
`global_k_binder_correction_l1` 从 epoch 1 的 `0.00517` 降至 epoch 2 的
`6.0e-6`，epoch 3 也只有 `8.7e-5`；parent/corrected entropy 几乎相同。
同时 object-innovation pair cosine 仍为负，故不能称为 global-K 全体同质坍塌。

Schema32 已联合记录 centered residual、parent/corrected top-2 margin、residual-to-margin
ratio 与 realized assignment change，不再用单一 L1 判断权力。剩余关闭条件是：
冻结 checkpoint 将校正器置零，依次比较 `GroundedFactSet -> S/W -> action`。若事实与
动作都近乎 bit-exact，应删除冗余校正而不是放大 residual；若事实变化而动作不变，归入
下游使用问题。

## O-07：P1 dynamic action self-write 可能压过缓存的高分辨率事实

**类型：V120 祖传结构风险；是否伤害 action 尚未由冻结干预证明。置信度：中高。**

源码锚点：`clearvla/mainline/model/policy.py::ClearVLAMainlinePolicy.velocity` 与
`clearvla/mainline/model/restored_bottom.py::RestoredV120EvidenceBottom.complete_p1_fact`。

当前仍保留 V120 P1 代数：

```text
canvas        = action_query + protected_detail
dynamic_delta = P1_policy_block(canvas) - canvas
completed_P1  = protected_detail + dynamic_delta
```

Schema30 epoch 3 的 `protected_detail≈0.0354`、`dynamic_delta≈0.2367`，后者约为
`6.7x`。这只是尺度风险，不能由此推断高分辨率事实已经丢失；P1 reader、24 factual
queries、N=49 与 3×3 microgrid 均未在 Schema32 改写。本项与信息流账本 IF-06 是
同一风险的主线释放门，不是第二个独立故障。

关闭条件：同年龄比较 V120 的 protected/dynamic/self/FFN RMS；冻结 checkpoint 做
detail zero/shuffle 与 action-query shuffle，先看 completed P1，再看 action。无链式证据前
不拆 P1、不缩减高分辨率读取。

## O-08：bottom neutral generic trajectory 可被映射为 trainable constant source

**类型：V120 祖传确定性常量 source；实际采用程度未知。置信度：高。**

源码锚点：`clearvla/mainline/model/restored_bottom.py` 的
`_neutral_trajectory_memory`，以及
`clearvla/mainline/v120_core/time_domain_mmdit.py::EvidenceViewAdapter`。

对象主路把 `owned_trajectory_memory` 置为精确零，但 trajectory 的
`LayerNorm + Linear` 含 bias，因此零输入仍可成为非零、数据集级常量 evidence。它可能是
合法 null，也可能吸收本应由 G/P1/W 提供的注意力；普通日志没有 source-level JVP，无法
判断采用程度。

Schema32 已记录 projected trajectory summary norm 和四个 basis 的实际 source mass；
剩余关闭条件是冻结 action JVP。若只承担 null，value
应精确为零且 null 身份在 value 外表达；若确有独立收益，需要可观测输入来源。不得用负
bias 或 quota 强迫少读。

## O-09：learned flow 的几何质量相对 Schema25 尚未恢复

**类型：已观测训练问题；与顶层问题可并存但不能生造单一因果。置信度：高。**

同 epoch 3，Schema25 的 warp/cycle/confidence 为
`0.09514/0.02468/0.2570`，Schema29 为 `0.09980/0.03526/0.2013`，Schema30 为
`0.09743/0.03357/0.2180`。Schema32 没有改 flow 模块或几何 loss，只有上游消费路径会
改变 action 梯度几何，因此新长跑前不能宣称本项已修复。

关闭条件：同 iter 比较 native/learned、warp/cycle/smooth/uncertainty、flow
magnitude/confidence、G geometry variation 与 P2 geometry posterior；必要时独立审查 flow
优化，不把它自动归因给 S/W。

## O-10：后期 tail/gripper 回弹仍未归因

**类型：完整训练泛化问题。置信度：高；结构独占因果未知。**

V120 与多个恢复 schema 都出现过早期下降后中远程或 gripper 回弹。P1 self-write、Teacher
目标质量、flow 几何、event 稀疏和数据覆盖都可能贡献，现有证据不能把它归给单一模块。
Schema32 改善的是确定的信息流闭环，不等于自动消除后期泛化问题。

关闭条件：Schema33 完成八轮；同时看 train/val action、first/tail、四 horizon bands、
arm/gripper、event/motion 与 condition-keep 分层。不能用 best checkpoint 或 batch 2200
代替全程。

## O-11：边界闭环尚未被证明为最终动作收益

**类型：可观测性债务。置信度：高。**

Schema32 的源码和合成测试能证明：canonical G content 同时进入监督和消费者、单一
typed S→W 入口、W common/residual 都经过 W-owned blocks、同一监督场进入 P2、真实
camera mixture、P2 common 不可被 null 丢弃、residual 保留精确零 null，以及
`effect -> consequence -> P3/bottom` 接线连续。这些证明的是结构正确性，不是动作净收益。

关闭条件：用同一冻结 Schema33 checkpoint 分层 zero/shuffle，并按
`source boundary -> W field -> P2/consequence -> bottom source -> action` 报告效应和置信区间。
只有边界与最终 action 都离开零，才声称策略使用该信息。不同 schema 同名但操作数改变的指标
不得直接做数值排名。长跑还必须证明 canonical slot/public-position capacity 实际获得梯度并
降低 reconstruction，而不是只把新增参数保留在零初始化；否则按优化/可识别性重新开户，
不能把“源码有路径”当成能力已经恢复。

## 统合后的依赖关系

- Schema32 已从源码层关闭 private reconstruction、W typed 调制瓶颈、W common/重复 S
  target 旁路和虚构跨相机坐标；此前关闭的 S common 淹没 residual、typed CoarseAction
  重复入口、successor/semantic 重复目标、diffuse Teacher 全权平均和 P2 null 丢弃公共 W
  仍保持关闭。
  它们若在新日志中仍以新指标复现，必须按新张量语义重新举证，不能沿用 Schema30 结论。
- O-01（G 校正）、O-07（P1 self-write）、O-08（bottom 常量）、O-09（flow）相互独立；
  没有干预证据前不得统一归因给 S/W。
- O-10 是完整曲线放行门，O-11 是因果放行门。两者未关闭前，不能宣称已经超过 V120。

## Schema33 后续检查

- fresh smoke：五步部署、Teacher 零调用、dtype/finite、参数 inventory；
- 同 iter 对比 V120/Schema25/Schema30 的 action、G/S/W/P、flow 和三阶段梯度；
- batch 2200 检查 common/residual target、prediction、P2 read 与 consequence 是否沿链条变化；
- 同时检查 canonical slot/public-position RMS 与 reconstruction、W typed-by-base interaction、
  W1/W2 common processing、P2 camera support/mixture，确认新增边界没有保持近零空转；
- 完成八个 epoch 后判断 O-10；冻结 checkpoint 后判断 O-01/O-07/O-08/O-11；
- 若结构边界健康但 action 无收益，归为数据/可识别性问题，不继续叠结构补丁。
