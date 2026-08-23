# ClearVLA 当前主线纯问题账本

更新：2026-08-23

当前源码身份：Schema34 `object_intent_dynamics_323`。行为比较锚点仍是 V120
`long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 与本地快照
`.audit/v120_exact_source_0b92d359/`。V120 是行为锚点，不是正确性公理。

Schema34 已从源码和可执行测试关闭 Schema33 的四项确定性问题：W common/residual
交叉公共化、public-S-only 时间所有权、P2 typed owner 错接，以及日志把 retained ratio
误称 cancellation。它们只记录在
`00_CURRENT_ARCHITECTURE_CONTRACT.md`，不在本账本保留历史副本。Schema34 尚无正式
CUDA 日志，因此当前没有新的 P0 接线故障结论；fresh smoke 和同龄 early gate 仍是放行前提。

## 记账规则

- 日志证明问题是否活跃及量级；源码证明数据流、旁路和目标代数。
- 只有源码可直接证明的轴丢失、错误接线、非法默认值或生命周期错误才称确定性故障。
- 张量存在、梯度非零、loss 下降都不等于策略正在使用该边界。
- 不用 gain、quota、hard gate、熵/多样性目标、额外外部 loss 或人工梯度掩盖问题。
- 历史版本只辅助判断继承、回归或放大；本账本只保存当前仍未关闭的问题。
- Schema34 必须 fresh run；Schema33 及更旧 checkpoint 不允许 exact resume。

## O-13：P2 complementary value 的单位适配效果尚待正式日志确认

**级别：P1。类型：继承性数值/容量适配风险；确定的无界单位旁路已关闭，实际字段贡献仍待观测。**

Schema33 的 common selected-value RMS 为：

```text
semantic 0.0564 / geometry 0.0287 / status 0.2340
```

原 protected `sum/sqrt(3)` 假定三个投影已经处于可比较张量年龄，导致 status 可仅凭原生
单位长期主导。Schema34 在每个候选 soft read 之前加入同一 one-sided、zero-preserving
`0.35/sqrt(3)` RMS 上界；它只衰减超界值，不放大弱值，不引入 type gate/配额，也不改变
all-null 精确零语义。这已经关闭“单一字段可无限凭单位占满 protected sum”的源码缺陷，
但不能证明 geometry 应与 status 等幅，也不能凭结构制造缺失的信息。

删除条件：fresh 日志同时报告每类 projected candidate RMS、contract scale、selected-value
RMS 与进入 consequence 的实际贡献。若高 status 来自超界单位，scale 应低于 1 且贡献差距
缩小；若 geometry 仍弱而 scale 为 1，应归为上游信息/任务可识别性，不得通过放大 geometry
或削弱 status 伪造平衡。冻结 action intervention 再量化净收益。

## O-14：Teacher partial assignment 仍过度分散，是未来差异监督的继承性上限

**级别：P1。类型：Teacher 可识别性/数据几何上限，不是 Schema34 新回归。**

Schema32/33 的 dustbin/null probability 约 `0.46–0.51`、effective support 约
`34–40`、reliability 约 `0.24–0.25`，`best_minus_background` 约 `-1.5` 到
`-2.0`。Teacher 确实输出 nontrivial residual，reliability 只作诊断/校准且不缩小 loss mask，
所以这里不是 reliability shortcut；但分散关联会限制细粒度 future effect 的监督上限。

删除条件：先看 Schema34 是否恢复 Teacher→W→P2 residual 保留。若 consumer 已恢复而
Teacher→W 仍是主要瓶颈，再单独研究 association；不得预先 sharpen temperature、强制
非零 flow 或用 reliability 加权 target。

## O-01：global-K 有界校正器的实际 assignment 权限偏弱

**级别：P2。类型：功能近空转风险；动作影响未知。**

G3 只校正 conditional-K 并保护 object-vs-null mass。Schema33 的 correction L1 约
`0.00405`、realized assignment change 约 `1.01%`，而 object content pair cosine
`0.530`、innovation pair cosine `-0.272`，说明对象没有整体同质化，但校正权限有限。

删除条件：冻结 checkpoint 将校正器置零，依次比较 `GroundedFactSet -> S/W -> action`。
若事实和动作均近乎 bit-exact，应删除冗余校正而不是放大 residual；若事实变而动作不变，
归入下游使用问题。本轮不修改 G。

## O-07：P1 dynamic action self-write 可能压过缓存的高分辨率事实

**级别：P2。类型：V120 祖传适配风险；动作伤害未证实。**

当前代数仍是：

```text
canvas        = action_query + protected_detail
dynamic_delta = P1_policy_block(canvas) - canvas
completed_P1  = protected_detail + dynamic_delta
```

Schema32/33 epoch 1 中 dynamic/protected RMS 比约 `8.2x/9.4x`。这可能是必要的动作条件化，
也可能覆盖静态精细事实；现有普通日志不能区分。P1 的 24 factual queries、N=49、3×3
microgrid 和单次高分辨率读取均仍完整，不能据此缩弱 P1。

删除条件：同一 checkpoint 分别干预 protected detail 与 dynamic self-write，沿
`P1 -> P2 -> action` 报告变化。确认有害后只处理 residual 适配，不压缩视觉读取。

## O-08：bottom neutral generic trajectory 可变成 trainable constant source

**级别：P2。类型：确定的常量接线，实际采用程度未知。**

对象主路把 generic trajectory memory 置为精确零，但 V120
`EvidenceViewAdapter.source_proj["trajectory"]` 含 bias，故零输入仍可成为数据集级常量。
它可能是合法 null，也可能吸收应由 G/P1/W 提供的选择质量。

删除条件：冻结 checkpoint 做 trajectory zero-bias/action JVP。若仅承担 null，value 应精确
为零并在 value 外表达 null identity；若有独立可观测收益，应改为具名来源。不得用负 bias
或 quota 强迫少读。本轮不修改 bottom。

## O-09：learned flow 几何质量偏弱且会受下游消费图反向扰动

**级别：P1。类型：已观测训练问题；不能生造单一结构因果。**

Schema33 相对 Schema32 未改 flow 源码，却出现 warp `0.09732 -> 0.10048`、cycle
`0.04065 -> 0.04924`、confidence `0.22199 -> 0.16744`、entropy
`0.73948 -> 0.80198`。这说明下游消费图可通过普通 autograd 扰动 flow 优化，但不证明
flow 实现自身产生新 bug。

删除条件：Schema34 同 seed/iter 比较 native/learned、warp/cycle/smooth/uncertainty、
flow magnitude/confidence、G geometry variation 与 P2 geometry posterior。若随 consumer
修复恢复，则并入本轮关闭；若仍落后，再独立审查 flow，不加非零 quota 或 identity 反约束。

## O-10：后期 tail/gripper 泛化回弹已确认但未归因

**级别：P1。类型：完整训练泛化问题。**

父运行在 epoch 6 达到 physical RMSE `0.08015`，epoch 8 回到 `0.08236`（约
`+2.75%`）；gripper physical 从 `0.1395` 回到 `0.1487`（约 `+6.6%`），同期训练
action loss 继续下降，train/val gap 持续扩大。这是真实泛化错位，不是暂时平台。

删除条件：Schema34 完成八轮，同时比较 train/val action、first/tail、三 horizon bands、
arm/gripper、event/motion 与 condition-keep；不能用 best checkpoint 或 batch 2200 代替全程。

## O-11：结构闭环尚未被证明为最终动作净收益

**级别：P1。类型：因果放行门。**

源码和合成测试只能证明 canonical G、typed S、W common/residual、P2、consequence 与 bottom
之间存在合法连续路径。Schema33 已证明 block update 和梯度均非零时，consumer 仍可能压缩
有效残差，所以“有路径”不能当作策略采用。

删除条件：冻结 Schema34 checkpoint 按
`source boundary -> W field -> P2/consequence -> bottom source -> action` 做 zero/shuffle，
报告边界效应、最终 action effect 与置信区间。只有两者都离开零，才声称策略使用该信息。

## 当前依赖与下一步

1. 先做 Schema34 fresh smoke，确认 five-step、Teacher 隔离、finite、参数 inventory、显存与
   throughput；不要 exact-resume Schema33。
2. 同龄 early gate 重点看 W residual/common、P2 retained/cancelled/support、public/typed/W
   temporal evidence、逐类型 candidate/contract/selected RMS、P3 effect/precision 与 flow。
3. 若 O-13 的单位旁路已关闭且 W→P2 带宽恢复，再完成八轮判断 O-10；否则只回溯出现问题的
   现有边界，不增加 block/loss/gain。
4. 冻结新 checkpoint 后判断 O-01/O-07/O-08/O-11。
5. 只有 consumer 正确而 Teacher→W 仍是主瓶颈时，才研究 O-14。
