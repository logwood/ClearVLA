# ClearVLA 当前信息流与适配问题账本

更新：2026-08-23

审查身份：Schema34 `object_intent_dynamics_323`。当前源码、完成的
Schema31/V31 日志、V120 `long` 行为锚点以及本地完整快照
`.audit/v120_exact_source_0b92d359/` 已交叉核对。

本文件不是架构契约，也不保存已经解决的问题。每个条目只描述当前仍未解决的
信息流或适配问题；满足删除条件后直接删除整个条目。禁止通过增加 gain、quota、
hard gate、熵目标、人工梯度或削弱健康旁路来伪造关闭。

Schema32 已关闭并从本账本删除的四项确定性故障是：私有 current-DINO 重建容量、
W 的窄 typed 调制、W common/重复 S 目标旁路，以及 P2 的虚构跨相机均值坐标。
它们的现行边界只记录在 `00_CURRENT_ARCHITECTURE_CONTRACT.md`。

Schema34 已关闭 W2/P2 的确定性 owner/时间证据接线和日志语义问题。P2 value 单位适配的
正式日志确认与 Teacher 分散上限由总账 `CURRENT_MAINLINE_ISSUES.md` 的 O-13/O-14 管理。
本文件不以另一组 IF 编号复制这些问题；IF-05–IF-08 只保留不同的独立适配风险。

## IF-05：S→P1 factual dock 将完整 Goal/History 再压成 mean/last 摘要

**级别：P2。类型：容量适配限制，不是硬断线。置信度：中。**

源码锚点：`clearvla/mainline/model/types.py` 中 factual dock 的构造。

S 内部已经用四个 interval queries 读取完整 Goal tokens 和完整 typed history；P1 的
`phase_context` 保留该 interval-specific 结果。但同一 factual dock 的独立 goal context
被压成四个 Goal query 的 mean，history context 只取最后一个 causal token，再复制到四个
interval。当前简单单任务可能被 `phase_context` 覆盖，因此不能把它称为当前动作退化的
确定原因；在多任务、相似终态或依赖较早动作历史时，它可能限制 P1 地址查询。

**删除条件：**先用冻结边界干预证明完整 Goal/History 相对现有摘要在 P1 address 上有
独立信息；若有，再用 query-based compact read 替代 mean/last，并保持 P1 24-query、
N=49、3×3 microgrid 和单次视觉读取不变。若无独立信息，记录证据后删除，不加容量。

## IF-06：P1 dynamic self-write 的尺度可能覆盖静态高分辨率事实

**级别：P2。类型：V120 祖传适配风险。置信度：中高；动作伤害未证实。**

源码锚点：`clearvla/mainline/model/policy.py` 与
`clearvla/mainline/model/restored_bottom.py::complete_p1_fact`。

P1 静态 reader、24 factual queries、N=49 posterior 和 3×3 microgrid 均完整；没有发现
视觉断线。但动态边界仍执行：

```text
canvas = action_query + protected_detail
dynamic_delta = block(canvas) - canvas
completed = protected_detail + dynamic_delta
```

既有日志中 dynamic delta 可达到 protected detail 的约 `6.7x`。这可能是必要动作条件化，
也可能覆盖静态事实；没有冻结干预前不能定性，更不能直接缩小 P1。

**删除条件：**同 checkpoint 分别干预 protected detail 与 dynamic action self-write，沿
completed P1→P2→action 报告变化。确认伤害后只调整残差边界或归一化适配，不缩减 P1
视觉读取；确认是有效条件化则记录证据并删除。

## IF-07：bottom 的精确零 trajectory 会被含 bias 的投影变成 trainable constant evidence

**级别：P2。类型：确定的常量接线，实际采用程度未知。置信度：高。**

源码锚点：`clearvla/mainline/model/restored_bottom.py::_neutral_trajectory_memory` 与
`clearvla/mainline/v120_core/time_domain_mmdit.py::EvidenceViewAdapter`。

对象主路将 generic trajectory memory 置为精确零，以避免复制 protected consequence；
但 `source_proj["trajectory"]` 是 `LayerNorm + Linear(bias=True)`，所以零输入仍会成为
数据集级常量 source。它可能是合法 null，也可能吸收本应由 G/P1/W 提供的选择质量。

Schema32 已记录 `evidence_trajectory_summary_norm` 以及四个 action-basis 的 trajectory
source mass，因此投影值和采用程度不再被合并指标遮蔽；尚缺冻结 action JVP，所以本项
不能仅凭普通训练日志关闭。

**删除条件：**补齐同 checkpoint 的 trajectory zero-bias/action JVP。若它只承担 null，
value 必须精确为零并把 null identity 放在 value 外；若确有独立收益，则改为明确的可观测
来源。不得用负 bias、quota 或硬门控强迫不读。

## IF-08：G pre-binding 用固定等权 logit consensus 融合统计性质不同的三类证据

**级别：P1。类型：可观测的适配风险，不是 type 轴断线。置信度：中高。**

源码锚点：`clearvla/mainline/model/grounding.py::_competition`。

semantic 与 appearance candidate view 都包含完整 DINO content；geometry view 主要包含
coordinate 与 learned geometry。当前将三类 K+null logits 固定算术平均后只做一次 physical
owner softmax。这保留了一个物理 K identity，但隐含三类证据同尺度、同可靠且统计独立；
实际上 semantic/appearance 可能重复计票，geometry 处于不同分布。Schema31 中
semantic-appearance posterior L1 通常约 `0.11-0.16`，另两组可约 `0.51-0.60`。Schema32
canonical decoded content 会让这项假设更值得监控，但没有因果证据前不改变融合代数。

不能把它改成另一个 outer type softmax，也不能让三个 type 各自拥有一套 K identity。
若确认错误绑定，应只使用当前可观测证据产生、构造上有界的 per-candidate reliability
校准 log-likelihood，并保证共享 content 只计一次。

**删除条件：**冻结 checkpoint 对三类 view 分别 zero/shuffle，报告 physical owner change、
G3 assignment change 和 action。若固定 consensus 无害，记录证据后删除；若有害，完成有界
校准并验证 single-K identity、null mass、camera/object permutation 均未破坏。

## 已核对但暂不记为故障的边界

- P2 query 是 `action_query + completed P1 fact`，不是未读取 P1 的 noisy-action query。
- S/W/P2 的 K、interval 和 semantic/appearance/geometry type 轴仍存在；三类 P2 value
  是互补分支，不再由外层 softmax 互斥竞争。
- P1 高分辨率读取与 transition 的完整 512 个 G3 rows 均保留；CVAE、workspace、
  Evidence MMDiT、transition 与 execution 主路未被顶层修复删除。
- Teacher 是 training-only、no-grad、每训练 batch 一次且部署零次；没有 future leak。
- uncertainty/reliability/covariance 不直接成为 P2 value 是 anti-shortcut 边界；其梯度
  不能当作 action-relevant W 使用证据。
- bottom 的直接 observable state/history 是合法 V120 路径。修复目标是让 W 提供互补
  后果，而不是破坏健康条件来强迫使用 W。
