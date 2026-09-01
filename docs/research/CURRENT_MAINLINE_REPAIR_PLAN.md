# ClearVLA Schema28→29 行为闭环与单/多任务合流计划

状态：**完整 Schema28、正式 Stage A attribution 与 estimator/full-proposal 门已完成；Schema29 detached action self-conditioning 已实现并通过本地闭环；八任务 gripper threshold 语义仍未闭合，Pen/RDT 联合 CUDA smoke 待完成**
更新：2026-09-01

本计划落实
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md) 中的完整八轮结论，
并规定单任务核心线与 `/data/rdt-ft-data/` 多任务接口线的合流顺序。目标不是
再跑一个低信息量试验，而是在同一个最终核心提交上让单任务回答网络闭环、
多任务回答外层接口与跨任务稳定性。

## 一、证据层级与当前决定

### 已确认实现事实

1. 训练只消费 `W(coarse)` 并调用一次 `velocity`；部署最终策略消费
   `W(proposal)` 并完成第二遍 ODE。
2. W 已经通过 P2 consequence 到达 ControlledTransition 和 Bottom。
3. CT 不是 world producer，也不是零梯度死路。
4. semantic 已承担强远端动作责任；geometry 的 learned-scale 动作责任弱。
5. Schema28 的 outer refinement 有非零 action/W change，但 final mismatch 与
   correction 同量级。

### 已确认行为事实

- final full/arm/gripper physical RMSE 为
  `0.07657 / 0.05677 / 0.14733`；
- final horizon bands 为 `0.02502 / 0.05513 / 0.09743`；
- decoded gripper event recall `0.2749`、ratio `0.4576`；
- W2 semantic/Teacher 约 `0.69x`，transport/Teacher 约 `0.44x`；
- 12 次 finite spike 主要仍由 observation owner 产生。

### 因果上仍未知

- train/runtime action-condition 错位是否是远端与 gripper 的主要原因；
- geometry 弱是合理窄角色还是下游过滤；
- 一次训练侧 detached self-conditioning 是否足以逼近部署 proposal 分布。

正式 attribution 已排除 W/CT 全局重复：W neutral 与 CT neutral 都有独立动作
增量，组合影响更大；W dynamic 与 consequence neutral 的 bit-exact 恒等同时证明
当前 sole-consumer 图完整。训练侧 self-conditioning 随后通过 estimator 匹配门并
成为活动 Schema29 源码；它的正式行为收益仍必须由新实验回答。

## 二、工作区与合流边界

只使用以下两条正确 lineage：

```text
codex/schema25-r1-replay
  semantic tip 90557b6
  owns: Schema28 core, evidence ledger, matched attribution

codex/rdt-multitask-prep
  base 76caa48 + RDT external preparation + multitask outlet + 90557b6
  owns: hierarchical data/language/camera/action adapter, multitask experiment
        entry and the same matched-attribution core
```

仓库根目录的旧 Schema39 branch 不属于本流程，不作为 donor、merge base 或
当前架构解释来源。

RDT 线已经完成算法外部的 bounded real-server loader acceptance 与本地多任务
experiment outlet；其权威设计为
[`auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md`](auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md)。
它当前证明外部数据 ABI、task-first sampling、逐任务 validation/logging 与单任务
行为不回归；尚不证明 RDT mixed-model CUDA smoke、三相机模型 consumer、双臂
codec/loss 或正式训练行为。

合流规则：

1. core attribution 已先在 replay 线成为独立语义提交；任何被放行的下一 core
   unit 仍必须遵循同一规则；
2. attribution 已合入 `rdt-multitask-prep`，数据域改动没有反向进入单任务基线；
3. 合流线同时保留 Pen 单任务入口和 fail-closed RDT 多任务入口；
4. 两个正式实验仍必须打印相同 core source/component digest；只能 dataset、adapter、
   task mix 和相应 action/camera profile 不同。

## 三、阶段 A：运行已实现的 validation-only matched attribution

这一步不训练、不写 checkpoint、不改变 primary deployment，也不增加 train-step
开销。它复用 Schema28 final checkpoint、同一批 validation 样本、同一 initial
physical noise 和同一 ODE 调度。

### A1. 先完成第一边界定义

每个 intervention 在写代码前必须列明：

```text
producer -> exact intervened tensor -> first consumer
retained axes / dtype / scale / zero semantics
expected first-boundary delta
action/horizon/gripper readout
```

特别禁止把 execution controller 的 `neutral` 当作 CT neutral，也禁止把
`protected_consequence` 整体清零后声称只移除了 W effect，因为其中包含 factual
base。

### A1.1 已完成的 active-source 正向数据流映射

```text
ObjectWorldBelief + PhysicalActionCondition[B,4,14]
  -> W physical projection + W1/W2
  -> CandidateWorld(
       same action-condition object,
       semantic [B,4,K,D],
       transport/covariance [B,4,K,C,2|3],
       FP32 current object/camera support)
  -> P2 spatial selection
       semantic: K within each physical interval
       geometry: K*C within each physical interval
  -> P2 no-null interval terminal, independently per type
  -> semantic/geometry latent sum
  -> one caller-owned 0.35 RMS contract
  -> consequence fact-gated typed interactions
  -> protected_consequence = factual_base + effect + interaction
       |-> P3 temporal condition and protected base
       |-> CT context plus terminal-normalized action tokens
       `-> Bottom protected no-null basis reader

completed G3 rollout [B,512,H]
  -> CT protected selector, built once per observation
  -> per-ODE real and learned-neutral coefficient evaluations
  -> CT value = gain * (real-neutral) x learned basis [B,512,H]
       |-> Bottom transition evidence memory
       |-> 4-anchor event-context pooling
       `-> output evidence-token audit
```

W values and P2/consequence/CT/Bottom activations follow the runtime autocast
dtype; support/log-support, covariance, coordinate scoring, normalization
denominators and matched-error accounting remain FP32 at their named boundaries.
W has one `.35` physical-action carrier contract; P2 has one `.35` aggregate
effect contract. Protected consequence has no extra gain. Bottom reads protected
consequence outside optional routing; dynamic P1 precision joins the optional
P3 sum only under the inherited fixed `0.25` ingress scale. CT trajectory uses
the existing variance-floored affine normalization and its learned `delta_gain`;
the attribution must not alter either scale.

Normal deployment builds initial W once, runs five updates plus an endpoint,
rebuilds W once, and runs the same six dynamic calls again from identical noise.
Every Stage-A counterfactual starts from the already refined cache and repeats
only that second six-call pass. It never rebuilds G/S/static-P1/CT source or
Teacher and never changes the primary two-pass result.

### A1.2 已完成的反向、owner 与 checkpoint 映射

训练中的 W 有两条梯度来源：`future_dynamics/future_transition` 直接监督
`W(coarse)`，action/execution loss 则沿
`Bottom -> CT/P3/consequence -> P2 -> W` 回传。Consequence 还通过 Bottom 的
protected reader、P3 temporal 和 CT 三个下游获得 action 梯度；CT value 同时经
Evidence adapter、event context 和 action decoder 获得梯度。不存在 CandidateWorld
绕过 P2 的 consumer，也不存在 W hidden 或 W value 直接写入 CT。CT 的 G3 selector
独立到达 Bottom 是保留的事实路径，不能在 `controlled_transition_delta_neutral`
中一并清零。

参数 owner 分别是 `dynamics`、`p2_effect_reader`、`consequence`、
`p3_compiler`、`controlled_transition`、`bottom_evidence_adapter`、
`bottom_policy_bridge` 和后续 decoder owners；每个参数只属于一个 AdamW group。
Stage A 只增加非持久 evaluation 状态和标量累计，不增加参数、buffer、optimizer
group、state key、RNG draw、loss 或 backward。Schema28 checkpoint 继续由
validation-only loader 读取；仅明确列入 validation replay allow-list 且由
primary-vs-explicit-none 测试证明无主路径差异的源码文件可以发生 source drift。
训练 exact resume 仍必须拒绝该 source drift。

### A1.3 源码审查后仍待实测的假设

1. 相同 refined cache、initial noise、dtype 和 eval 状态下，`explicit_none` 必须与
   primary bit-exact；否则所有 matched 结果作废。
2. `world_dynamic_neutral` 与 `consequence_effect_neutral` 干预点不同，但根据当前
   sole-consumer 图应产生完全相同的下游动作。保留两者一次用于检验未映射旁路；
   若恒等成立，后续日志/实验可删除其中一个重复模式。
3. `controlled_transition_delta_neutral` 必须令 value 精确为零、real coefficients
   精确等于 learned-neutral，同时保持 G3 selector、context 和 action-token计算不变。
4. `wrong_action_world` 只轮换 batch 内 interval action，保留当前样本 action-state
   anchor、belief、support 与 camera geometry。batch 小于 2 或 donor action delta
   为零的行不能提供 identifiability 证据，必须显式记为无效而不能伪造覆盖。
5. 默认 16 个分散 batch 是否含足够 episode、gripper event 与非零 donor delta
   仍需由结果中的有效 batch/row 计数决定；只有这些计数会改变决策时才增加预算。

前四项已经由当前源码回归和一批完整 CPU FP32 validation smoke 在 fresh 小模型上
通过：`explicit_none` 与 primary、`world_dynamic_neutral` 与
`consequence_effect_neutral` 均 bit-exact；CT 网络仍执行，value 为 exact zero，
action coefficients 等于 learned-neutral，G3 selector 不变；wrong-world 只替换
interval action 并保留接收样本 anchor/belief/support/camera。正式 Schema28
checkpoint、BF16/CUDA 数值和第 5 项覆盖仍未实测，因此这些本地恒等不能被解释为
行为 attribution 结论。

### A2. 最小配对集合

在同一固定 subset 上实现以下语义模式：

1. `primary`：普通 Schema28 refined deployment；
2. `world_dynamic_neutral`：保留 current belief、support、action tag 与 factual
   base，只将 CandidateWorld 的预测 dynamic effect 置为其代数 neutral；
3. `consequence_effect_neutral`：保留 factual base 和 dynamic P1 precision，只移除
   P2 typed effect/interaction 对 consequence 的增量；
4. `controlled_transition_delta_neutral`：保留 G3 source、context、action tokens 与
   learned-neutral定义，只移除 real-minus-neutral transition delta；
5. `world_dynamic_neutral + controlled_transition_delta_neutral`：用于判断两者的
   加性/交互责任；
6. `wrong_action_world`：从同 batch 的确定性 donor action 重建 W，保留当前样本
   的其他路径，验证 W 对正确动作条件的特异性。

如果源码审查证明其中两个模式在第一边界上同义，则删除重复模式，不用别名
增加表面覆盖。默认仍使用现有 16 个 diagnostic batches；只有 episode/event
覆盖不足以改变决定时才扩大，不扩到全验证集。

### A3. 必须记录的最小结果

- 第一边界 intended delta 和所有非目标边界 identity error；
- full/arm/gripper、three horizon bands、decoded event ratio/P/R/F1 及该 subset 的
  predicted/target event counts；
- proposal/refined delta、final interval/delta mismatch；
- W semantic/transport change、consequence effect、CT value/action delta；
- paired MSE gain 与 paired action delta，不增加 tensor dump。

若第一边界没有改变、coverage 不完整或 primary-vs-explicit-none 不一致，干预
结果无效，不能进入结构决策。

当前实现的 mainline/runtime/checkpoint/auditor 相关选择通过 `223/223`；另一个
真实执行全部六种 core modes、既有四种 P2 modes、proposal/execution ablations 的
一批 fresh CPU FP32 validation smoke 通过。

### A4. 正式 Schema28 checkpoint 结果（已完成）

正式 replay 使用 Schema28 final checkpoint、179 个 validation batch 与分散的
16-batch attribution subset，运行 `815.015 s`。有效性门全部通过：

- primary 与 `explicit_none` normalized/physical action 均 bit-exact；
- `world_dynamic_neutral` 与 `consequence_effect_neutral` 均 bit-exact；
- wrong-action donor `128/128` 行有效，subset 内 target event `117`，不是空干预；
- W semantic/transport 第一边界变化为 `0.24847 / 0.04108`；
- 去掉 W dynamic 后 13--24 action delta `0.05829`、gripper delta `0.13240`，
  paired MSE 分别恶化 `0.00557 / 0.03203`；
- 去掉 CT delta 后对应 action/gripper delta `0.01605 / 0.04142`，MSE 分别
  恶化 `0.00094 / 0.00655`；联合 neutral 的 delta 为 `0.06527 / 0.15345`；
- wrong-action W 的首边界 action-condition/semantic/transport delta 为
  `0.13604 / 0.02639 / 0.00406`，13--24 action/gripper delta 为
  `0.00943 / 0.01952`。该 donor 在此小 subset 上误差略有改善，因此它只证明
  correct-action identifiability，不证明当前条件已经最优。

据此 Stage B 选择决策表第一行：保留 W 与 CT 的独立职责，只允许继续检查训练侧
action-conditioned W 分布对齐。geometry gain、W->CT bridge、CT world generator、
transport quota 与 hard event gate 继续禁止。

## 四、阶段 B：用 attribution 选择唯一核心语义单元

### 决策表

| 结果 | 下一步 |
|---|---|
| W/wrong-world 明显改变远端动作，CT 也有独立增量 | 保留两者职责，优先修训练侧 action-conditioned W 分布 |
| W effect 有表示变化但 action delta 近零 | 先审 P2/consequence/Bottom 过滤，不增加 W->CT bridge |
| CT neutral 近零而 W effect 有强责任 | 完整审 CT consumer 后才考虑收缩重复职责，不直接删除 |
| W 与 CT 单独都弱、组合明显 | 审查交互/尺度竞争，不靠增益制造单路依赖 |
| wrong-world 与 correct-world 无可辨动作差异 | self-conditioning 不放行，先关闭 W identifiability |
| geometry neutral 仍近零而 semantic 强 | 保持 geometry 窄角色，不放大；后续用多相机任务再判断 |

任何分支都不允许重新采用 W->CT bridge、CT world generator、geometry gain、
transport quota 或 hard gripper event gate。

## 五、阶段 C：已实现——训练侧 detached action self-conditioning

阶段 A 已证明 W 对正确 action/world 有动作责任。本节修调用顺序，
不增加第二套网络、不增加外部 loss weight，也不声称完全复现五步 proposal ODE。

### C1. 候选前向

```text
encode observation/G/S/static-P1 once
  -> cache0 owns W(coarse)

same sampled t and same noisy physical field
  -> velocity pass0 under no-grad
  -> clean_physical0 = noisy + (1 - t) * velocity0
  -> codec.decode(clean_physical0.detach(), action_state)
  -> existing PhysicalActionCondition.from_horizon_action
  -> rebuild only W -> cache1

same t and same noisy physical field
  -> velocity pass1(cache1)
  -> existing action/event/motion/execution losses
  -> existing W future loss reads cache1.predicted_dynamics
```

第一遍只产生 detached condition；参数更新来自第二遍以及既有 static top losses。
两遍共享同一个模型参数。不得给 pass0 新增 auxiliary action loss，也不得把两遍
loss 简单相加从而翻倍 action budget。

### C2. 明确的近似边界

训练 pass0 是随机 flow-time 上的 clean endpoint estimate；deployment proposal 是
完整五步 ODE。二者不是 bit-exact 分布匹配。接受这一候选前必须离线比较：

- train estimator 与同 checkpoint 完整 proposal 的 action-condition RMS/方向；
- 二者重建 W 后 semantic/transport 差异；
- 额外 dynamic call 的 CUDA memory/throughput。

若 estimator 与完整 proposal 的差异大到改变 matched W 责任，停止本候选；不把
第三遍 ODE 塞进训练来强行追平。

这项门由 validation-only observer 实现：每个既有 diagnostic batch 复用完整
proposal 已拥有的 initial physical noise，按训练 flow-time 分布取一个 `t`，只执行
一次 cache0 endpoint velocity，detached decode 后只重建 W。它记录 normalized
interval action/delta RMS、相对 coarse baseline 的 ratio、更新方向 cosine/有效覆盖、
semantic/transport W RMS，以及完整额外路径的时间/实时 CUDA allocation。它不替换
primary sample，不进入 loss，不增加参数、buffer、optimizer/checkpoint state 或全局
RNG draw。observer 在 eval mode 下运行，因此训练 dropout 的匹配随机流仍属于真正
Schema29 实现 smoke 的独立合同，不能由本门代替。

正式 gate 已在 179 个 validation batch、16 个 diagnostic batch 上完成：estimator
相对 coarse 的 full-proposal interval action/delta 距离为
`0.210828x / 0.115498x`，semantic/transport W 距离为
`0.168057x / 0.221124x`，更新方向 cosine `0.984706`、有效覆盖 `1.0`；额外路径
开销 `0.200390 s/diagnostic batch`、live allocation `0.013094 GiB`。该结果放行
Schema29，但不预判正式训练收益。

### C3. 梯度与生命周期合同

- pass0、decode、`PhysicalActionCondition` 对 pass1 条件的路径必须 detached；
- pass1 action gradient正常到 P2/consequence/CT/Bottom 和 W 参数，但不反传到
  pass0 的 sampled action；
- observation/G/S/static-P1/Teacher 仍各构建一次；只重建 W 和完整 dynamic path；
- 使用同一个 `FlowMatchingState`，不新增 RNG draw；
- optimizer 参数集合不变；如无新参数，state-key 与 parameter inventory 不变；
- training/runtime ABI 与 manifest 必须升级，Schema28 checkpoint 不得 exact
  optimizer resume；只允许明确审计过的初始化/迁移策略；
- first-boundary、consumer-backward、checkpoint 和 deployment call-count 必须在
  修改后各自反向复核一次。

当前实现满足以上合同：一次 flow 采样、pass0/pass1 两次 velocity、pass0
`no_grad`、condition detached、cache1 同时进入唯一正式 action loss 与 future
loss；forked RNG 令两遍 dropout 入口一致且全局状态只留下正式 pass1 的推进。没有
新增参数、state key、optimizer owner 或 objective，Schema28 exact resume 被拒绝，
部署仍是两遍各五次 update 加 endpoint。相关完整选择 `124/124` 通过。

### C4. 节制诊断面

训练 JSONL 只新增或重用以下 decision-facing 标量：

- pass0 endpoint action RMS、pass0->pass1 clean-action delta；
- coarse->pass0 PhysicalActionCondition delta；
- W rebuild semantic/transport delta；
- pass1 clean action 与其 W condition 的 interval/delta mismatch；
- pass0 detached audit action loss（只观测）与正式 pass1 action loss的差；
- W physical-condition VJP、P2 semantic/geometry effect VJP、CT owner gradient。

不增加每 batch tensor、完整 probe dump、重复别名或 console 长矩阵。现有 100-batch
JSONL cadence 保持；console 只显示 stop/continue 所需摘要。

## 六、阶段 D：多任务接口合流，不分叉网络行为

多任务线先保持现有算法外部适配，随后只补正式 experiment outlet：

1. 任务 registry/manifest 明确列出首轮任务集合，不靠目录名推断目标；
2. 每个样本通过同一 canonical `TrainingBatch` 进入共享 core；
3. 逐任务输出 full/arm-or-joint-group/gripper、horizon 和有效样本数；同时给出
   micro 汇总，macro 只作为每任务等权摘要；
4. camera/action profile、task id、instruction identity、normalizer 与 sampling mass
   写入 run context；这些 metadata 不成为隐藏的模型旁路；
5. 首轮仍使用已经验收的 right-arm/two-camera profile 验证训练出口。原生三相机
   consumer 与 14-D bimanual codec/loss 是后续显式 ABI 单元，不能由 adapter
   偷偷截断后宣称“多相机/双臂已支持”；
6. Pen 单任务入口暂时保留以诊断核心；多任务入口稳定后，Pen 可作为单任务
   registry 配置进入同一入口，再淘汰旧 loader，不淘汰核心行为测试。

## 七、无正式实验的联合 smoke

### 单任务 smoke

- 一个真实 batch，fresh forward/backward/optimizer step；
- pass0 detached、W rebuild、pass1 loss、finite gradient；
- checkpoint save/load 与 exact rejection/migration contract；
- 五步 proposal/refined deployment 和所有 matched attribution modes。

### 多任务 smoke

- 首轮每个任务各一个 batch，再做一个 mixed batch；
- episode/language/camera/action profile 与 task id 对齐；
- 每任务 loss/metric denominator 正确，缺失任务不输出伪零；
- 同一 Pen 样本经旧单任务 adapter 与新 registry 单任务配置后，进入 core 的
  tensor、mask、normalizer 和 language row 必须等价。

smoke 失败只修 ABI/实现，不用正式 GPU 训练判调试错误。

## 八、同一核心提交的两个正式实验

1. 先启动 Pen 单任务，完成 preflight 与首个健康窗口；确认不是立即 non-finite、
   lineage、memory 或 loss-ledger 故障后即可启动多任务，不等待八轮结束。
2. 两个运行串行占用同一 GPU；不得并发造成吞吐/显存不可比。
3. 单任务看 core closure：far horizon、gripper、W/CT、refinement mismatch、spike。
4. 多任务看 adapter/task competition：逐任务曲线、camera/action profile、sampling
   share 与跨任务梯度健康。

归因矩阵：

| 单任务 | 多任务 | 主要判读 |
|---|---|---|
| 坏 | 坏 | core unit 或共享训练引擎 |
| 正常/改善 | 坏 | adapter、task mix、profile 或跨任务竞争 |
| 改善 | 各任务稳定 | core 与外层接口同时通过 |
| 改善 | 少数任务坏 | 回到该任务的数据/action/camera 语义，不回滚整个 core |

## 九、正式放行与停跑规则

硬停：non-finite、lineage/identity failure、loss ledger 不闭合、目标边界 intervention
无效、重复 severe spike、显存超界或 checkpoint ABI 违规。

不会单独触发停跑：早期 event F1 低、geometry RMS 小、transport/Teacher ratio
未达某个固定值、capacity 接近全开。这些必须与动作责任和完整曲线一起解释。

单任务验收至少要求：

- aggregate、arm、gripper、first/tail、三段 horizon 不以近端换远端；
- event ratio/recall 与 post-event 三段同时看；
- final mismatch 相对 proposal->refined correction 不再恶化；
- correct-world 优于 wrong-world，且 W/CT matched 责任可解释；
- spike owner/count/max 与 Schema28 完整曲线比较；
- 完整八轮，不以 best epoch 代替 final。

多任务验收必须逐任务列出，不允许只给 aggregate。任何任务都要带样本/事件数、
有效 camera/action profile 和 normalizer；macro 只回答“典型任务”，micro 只回答
“总体样本”，二者不能互相替代。

## 十、当前执行顺序

```text
[done] 固化本问题账本与本计划
[done] 在 replay 线实现并本地验证 validation-only matched attribution
[done] 将 attribution 合入 rdt-multitask-prep
[done] 补 task-first sampler、task-stratified validation、逐任务/micro/macro
       日志和独立 RDT launcher
[done] 在 Schema28 checkpoint 上运行 validation-only matched attribution
[done] 明确 RDT continuous gripper：相邻 command 定义 activity/event，qpos 只作
       codec/物理解码边界
[next] 用 train-only adjacent-command audit 定义并显式采用一个有源码依据的共享
       raw gripper activity threshold；描述性分位数不得自动升级为阈值
[done] 根据 attribution 决策表选择 detached self-conditioning 进入 estimator 门
[done] 在同一 Schema28 checkpoint 上完成 estimator/full-proposal 匹配门
[done] 对被放行 core unit 做双向源码审查并实现 Schema29
[next] 在 Schema29 精确提交上完成 Pen 单任务 CUDA smoke
[next] 在相同精确提交上完成 RDT batch-eight mixed-model CUDA smoke
[last] 依次启动 Pen 单任务与 RDT 多任务正式实验
```

没有阶段 A 的有效因果边界，不进入阶段 C；没有单/多任务 tensor-equivalence 与
联合 smoke，不启动两个正式实验。
