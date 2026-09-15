# ClearVLA conversation insight index

Status: compact historical research index and frozen-endpoint synthesis,
completed through visible message 5493. The full chronological archive is retained
in Git history; this file is not an active architecture contract or implementation
authority.

This ledger records decisions, hypotheses, observations, reversals and durable
engineering lessons from the long-running ClearVLA conversation. It is organized
by conversation order rather than by the checked-out source version. Historical
messages are evidence about what was believed or attempted at that time; they are
not instructions to reapply old code.

Interaction context is recorded only when frustration, anger, apology or trust
materially changed a technical requirement, review method or decision boundary.
Such entries describe the trigger, immediate effect and later reflection without
treating tone as technical evidence, excusing implementation failures or judging
either participant's character.

Active architecture truth remains in
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md).
Current blockers remain in
[`../CURRENT_MAINLINE_ISSUES.md`](../CURRENT_MAINLINE_ISSUES.md). The source-level
V120-to-Schema39 reconstruction remains in
[`ARCHITECTURE_REPLAY_LEDGER.md`](../archive/replay/ARCHITECTURE_REPLAY_LEDGER.md).

## Card contract: frozen judgment and evidence

Every card is judged at the same frozen endpoint: visible message 5493. The
`later view`, judgment and evidence fields describe what the completed archive
supported at that endpoint; none of them silently describes the current checkout.

`【冻结终点判定】` separates two axes that the original free-form statuses mixed:

- `结论处置` applies to the historical claim, attribution, proposal or
  implementation named by the card:
  - `保留`: the scoped historical conclusion survives;
  - `部分保留`: only the stated subset or principle survives;
  - `已否定`: that historical subject was contradicted, withdrawn or excluded;
  - `已吸收`: it survives only inside a later design and is not an independent
    recovery unit;
  - `待证`: the proposed mechanism or causal attribution remains unresolved.
- `证据闭环` states how far the scoped judgment was established:
  - `已闭环`: direct historical evidence resolves the scoped claim;
  - `部分闭环`: an observation or local boundary is established, while an
    isolated cause, benefit or downstream effect remains open;
  - `未闭环`: the archive contains a proposal or attribution but not its required
    direct test.
- `说明` preserves the original qualifier so implementation repairs, replacements,
  rejected subclaims and remaining limitations are not flattened by the enums.

These axes deliberately permit combinations that a single status word could not
express. For example, a suspected root cause can be `已否定 / 已闭环`; an
observed spike with an unknown origin can be `待证 / 部分闭环`; and a sound
principle whose isolated task benefit was never measured can be `保留 / 部分闭环`.
The disposition is about the card's historical subject, not whether the negative
lesson is worth retaining.

Across the 431 frozen cards, the normalized dispositions are `保留 307 / 部分保留
83 / 已吸收 22 / 待证 9 / 已否定 10`; evidence closure is `已闭环 300 /
部分闭环 127 / 未闭环 4`.

`【证据与锚点】` has a separate retrieval contract:

- `类型` is a multi-label index over source facts, log observations,
  tests/runtime, causal interventions, data probes, process decisions, historical
  retrieval, design records and explicitly unresolved inference. It is an index,
  not a proof-strength score.
- `对话` points to the exact visible-message range in the frozen raw rollout.
- `段级工件` points to the segment boundary's `Primary historical artifacts`.
  That list is the search/evidence scope for the segment, not a claim that every
  listed artifact directly proves every card.
- `直接标识` is present only when a commit, path, log, serialized field or
  similarly exact identifier occurs literally in the card. It is historical text;
  it may have moved or disappeared from the current checkout.
- `直接量值/表达式` separately preserves literal schedules, ratios, units or
  equations so that numeric evidence is not mislabeled as an artifact identifier.
- `补充直接证据` preserves any card-specific numeric or run evidence that
  was already separately anchored.

A card with only a conversation anchor and segment-level artifact scope must not
be upgraded into an independently reproduced source/log fact. Source-level V120
through Schema39 claims should also be cross-checked against the archived
[`ARCHITECTURE_REPLAY_LEDGER.md`](../archive/replay/ARCHITECTURE_REPLAY_LEDGER.md)
before reuse.

## Curated research map

这部分是索引的主入口。完整 431 张卡已从工作区移出并保留在 Git 历史中，不应
把每张卡都当作一个仍需修复的当前问题。

推荐阅读顺序：

1. 查当前实现，先读 `../00_CURRENT_ARCHITECTURE_CONTRACT.md` 和当前源码；
2. 查历史结论，先读下面的“规范化主结论”和“关键反转”；
3. 需要版本因果链时，再按“版本阶段地图”进入相应 Segment；
4. 需要证据细节时，最后搜索稳定的 `CI-xxxx` 卡片；
5. 任何历史恢复建议仍须在当前源码、run context 和当前日志上重新验证。

这里采用非破坏性去重：每条规范化结论只选少量“主卡”承载结论本身，其余“支持卡”
保留不同版本、不同层次或反证后的证据。重复出现不等于重复结论；若两张卡分别记录了
假设与推翻、producer 与 consumer、训练与验证，就不能为了缩短文档而合并或删掉。

### 规范化主结论

1. **真实执行图优先于版本故事。** 版本名、配置开关、类名、参数量和非零梯度都不能
   单独证明某条路径实际运行并影响动作。必须联合 launcher、序列化身份、producer、
   consumer、loss、optimizer、deployment 和日志核实。主卡：`CI-0001`、`CI-0002`。
   支持卡：`CI-0032`、`CI-0087`～`CI-0089`、`CI-0225`、`CI-0410`、`CI-0411`。

2. **信息所有权不等于信息排他。** 明确唯一 owner 是为了避免重复写入和旁路，不是为了
   删除 common、identity、camera、type 或 precision。多次失败都来自把连续的权衡翻译成
   二元禁令。主卡：`CI-0357`、`CI-0391`、`CI-0392`。支持卡：`CI-0155`、
   `CI-0390`、`CI-0416`。

3. **功能闭环必须到达动作消费者。** 一个字段被预测、监督、具有梯度或在日志中变化，
   都不能替代 matched intervention 和最终 action/gripper 后果。主卡：`CI-0158`、
   `CI-0162`、`CI-0395`。支持卡：`CI-0161`、`CI-0234`、`CI-0247`、
   `CI-0404`、`CI-0413`。

4. **轴必须保留到合法终端，不能先压缩再伪造回来。** K、camera、interval、type、basis、
   lane 和 action-query 是语义轴，不是恰好相等的 shape。同尺寸广播不构成语义恢复；
   冻结终点采用 P2 终结空间身份、P3 终结时间，但这项具体分工仍须以 active graph 为准。
   主卡：`CI-0131`、`CI-0167`、`CI-0215`。支持卡：`CI-0149`、`CI-0334`、
   `CI-0344`、`CI-0408`、`CI-0429`。

5. **protected 与 optional 必须严格分权。** null/dustbin 可以拒绝一条零值 optional
   innovation，不能擦除事实、完整 W future 或 protected consequence。validity、existence、
   allocation、reliability 和 loss support 也不是同一种概率。主卡：`CI-0197`、
   `CI-0253`、`CI-0288`。支持卡：`CI-0129`、`CI-0241`、`CI-0265`、
   `CI-0335`、`CI-0342`、`CI-0358`、`CI-0416`。

6. **互补证据不应争夺一份概率质量。** semantic、appearance、geometry、status 或多条
   innovation lane 只有在表达互斥假设时才适合共享 softmax。固定平均虽然避免竞争，也可能
   制造尺度损失；正确边界是保留身份，在合法 consumer 处组合并共享有界预算。主卡：
   `CI-0203`、`CI-0349`、`CI-0352`。支持卡：`CI-0350`、`CI-0373`、`CI-0421`。

7. **监督必须对应唯一在线字段和可观测语义。** successor/delta 必须满足同一代数身份；
   dustbin 不是物理消失；reliability 不是未来 loss mask；post-hoc predictor 不能替代 P
   实际消费的 W。主卡：`CI-0158`、`CI-0162`、`CI-0358`、`CI-0418`。
   支持卡：`CI-0177`、`CI-0243`、`CI-0289`、`CI-0310`、`CI-0342`、
   `CI-0346`、`CI-0422`。

8. **静态事实、动态策略和未来后果拥有不同生命周期。** observation/G/S/W/static P1
   在冻结终点属于 observation；dynamic P1/P2/P3/bottom 属于 ODE action query；Teacher
   只属于训练 target。缓存、重复 forward 或把 dynamic precision 冒充 factual base 都会
   改变梯度和行为。主卡：`CI-0176`、`CI-0272`、`CI-0283`、`CI-0384`。
   支持卡：`CI-0187`、`CI-0284`、`CI-0387`、`CI-0401`、`CI-0408`。

9. **forward 有界不代表 backward 有界。** near-zero normalization、平方根/方差、重复概率
   过滤、BF16 饱和、固定平均和 global clip 都可能形成隐藏 Jacobian 或梯度竞争。数值 measure
   应由 producer 以正确 dtype/units 导出，不能在 consumer 丢失后重建。主卡：`CI-0083`、
   `CI-0144`、`CI-0178`、`CI-0409`。支持卡：`CI-0037`、`CI-0114`、
   `CI-0141`、`CI-0255`、`CI-0256`、`CI-0381`、`CI-0384`、`CI-0414`。

10. **内部量的大小不是质量结论。** flow RMS、G3 residual、interval variation、cosine、
    optional RMS 和梯度都必须先统一单位、统计 support、owner 与对照。小值可能符合数据先验，
    大值也可能只是公共旁路。主卡：`CI-0063`、`CI-0103`、`CI-0308`。
    支持卡：`CI-0104`、`CI-0105`、`CI-0140`、`CI-0184`、`CI-0327`、
    `CI-0374`、`CI-0430`。

11. **训练、验证与部署必须共享生命周期契约。** 每个 dormant branch、warmup 边界、BF16
    组合、五步 sampler、endpoint head、checkpoint/resume 和 production startup 都必须被真实
    执行；局部测试通过只恢复了“有资格跑实验”。主卡：`CI-0011`、`CI-0015`、
    `CI-0040`。支持卡：`CI-0014`、`CI-0037`～`CI-0039`、`CI-0086`、
    `CI-0218`、`CI-0295`、`CI-0410`、`CI-0411`。

12. **日志与审计器是实验语义的一部分。** missing、inactive、zero、failed 必须分开；raw
    objective 与 weighted contribution 必须可重建；同 iter、同数据、同 normalizer、同 owner
    才能比较。主卡：`CI-0017`、`CI-0021`、`CI-0024`。支持卡：`CI-0018`～
    `CI-0020`、`CI-0022`、`CI-0025`、`CI-0085`、`CI-0112`、`CI-0192`、
    `CI-0244`、`CI-0245`、`CI-0249`、`CI-0305`。

13. **历史锚点有不同职责。** V120 是行为/回归/旧主路健康锚点但并不完美；V25/V26 是
    性能和 replay island；直接父版本只用于判断新回归。不存在一个可整体恢复的“完美健康版”。
    主卡：`CI-0412`、`CI-0420`。支持卡：`CI-0232`、`CI-0307`、`CI-0323`、
    `CI-0370`、`CI-0399`、`CI-0423`。

14. **持久文档也必须分工。** 当前契约只描述 active graph；当前问题账本只保存未解决问题；
    replay ledger 保存源码谱系；本文件保存讨论、证据和反转。压缩上下文会放大漂移，但不能
    代替源码审查。主卡：`CI-0151`、`CI-0186`、`CI-0424`、`CI-0428`。
    支持卡：`CI-0171`、`CI-0211`、`CI-0274`、`CI-0275`、`CI-0372`、
    `CI-0425`～`CI-0427`。

### 版本阶段地图

| 阶段 | Segment / CI | 核心问题 | 结论性质 |
|---|---|---|---|
| V93/V94 execution | 001–004 / CI-0001～0041 | controller、soft/hard、warmup、dtype、部署 | 建立实验与生命周期审查基本法 |
| V95/V96 top representation | 005–007 / CI-0042～0069 | pooled top、Flow-DINO/JEPA、Stage、长程 | 从单向量转为多时域空间证据；小 flow 结论被数据探针修正 |
| V97–V100 raw/address | 008–012 / CI-0070～0114 | 3+3+2、raw pyramid、Flow 地址、observable motion | 证明“有 raw/有梯度”仍可 action-irrelevant |
| V101–V108 causal localization | 013–017 / CI-0115～0150 | phase/space/detail 因果、late read、数值修复、progressive G | 形成轴保持、真实干预和正反向审查 |
| V109–V119 typed future path | 018–020 / CI-0151～0187 | G/S/W/P ownership、future effect、stage 可识别性 | 多次证明 sidecar supervision 与 action consumer 可分离 |
| V119→V120→V121/V122 / independent extraction | 021–024 / CI-0188～0232 | object/intent/dynamics、completion、独立 mainline | V120 成为行为锚点；V121/V122 与新 mainline 暴露回归 |
| Schema18–24 recovery | 025–032 / CI-0233～0306 | 完整源码恢复、optimizer/lifecycle、P1/Teacher/camera | 机械恢复仍须闭合 adapter、顺序和行为 |
| Schema24–32 ownership | 033–037 / CI-0307～0367 | S typed、G3、null、support、P2 type、W common/residual | V25/V26 形成性能恢复岛；局部“纯化”持续制造传递损失 |
| Schema33–39 closure | 038–042 / CI-0368～0424 | 时间终端、55×、W publicization、FP32 measure、历史回放 | 修正多层消费者，但 Teacher→W 时间带宽与 gripper 仍未闭合 |
| 记忆与方法收尾 | 043 / CI-0425～0431 | compaction、原始事件流、卡片方法、术语考证 | 将完整对话转为可复核而非可执行的研究记忆 |

### 子系统与问题族索引

| 要查什么 | 先读的主卡 | 说明 |
|---|---|---|
| 实验身份、launcher、checkpoint、resume | CI-0001、CI-0032、CI-0087～0089、CI-0180、CI-0223～0227、CI-0293、CI-0410～0411 | 先确认“跑的是什么”，再解释指标 |
| execution / sampler / train-deploy | CI-0006～0019、CI-0026～0028、CI-0037～0040、CI-0142、CI-0256 | 覆盖 warmup、soft/hard、五步与 endpoint |
| Flow、DINO、raw detail、spatial address | CI-0045～0048、CI-0063～0069、CI-0071～0080、CI-0096～0106、CI-0121 | 区分 correspondence、地址、value 和 action utility |
| G progressive grounding / K identity | CI-0127～0134、CI-0147～0150、CI-0194、CI-0230、CI-0239、CI-0291、CI-0294、CI-0311、CI-0329～0331、CI-0360～0362 | 重点看 rematerialization、consumer 和重建旁路 |
| S intent / temporal organization | CI-0137、CI-0143、CI-0166、CI-0176、CI-0179～0183、CI-0187、CI-0254、CI-0310～0317、CI-0332、CI-0340、CI-0345、CI-0373、CI-0405 | 区分合法 prior、typed owner 和伪阶段 |
| Teacher / W future dynamics | CI-0152、CI-0158、CI-0161～0165、CI-0176～0178、CI-0193、CI-0206～0208、CI-0243、CI-0289～0294、CI-0333、CI-0337、CI-0342～0348、CI-0358～0364、CI-0388～0395、CI-0413、CI-0418、CI-0422 | 重点看 target identity、dustbin、interval bandwidth 和 action marginal use |
| P1/P2/P3 / consequence / bottom | CI-0069、CI-0074、CI-0115～0122、CI-0146～0149、CI-0161、CI-0167～0170、CI-0185、CI-0193、CI-0202～0209、CI-0240～0242、CI-0253、CI-0279、CI-0283～0288、CI-0298～0299、CI-0334～0336、CI-0348～0355、CI-0364、CI-0377～0382、CI-0400～0408、CI-0421 | 查 protected/optional、终端轴和重复 carrier |
| 数值、梯度、尺度、显存 | CI-0037、CI-0058、CI-0077、CI-0082～0084、CI-0114、CI-0136、CI-0138、CI-0141～0144、CI-0156、CI-0159、CI-0169、CI-0178、CI-0255～0258、CI-0280、CI-0287、CI-0290、CI-0295、CI-0302、CI-0352、CI-0381、CI-0384、CI-0403、CI-0409、CI-0414 | 不把有限 forward、单参数峰值或低显存直接当根因 |
| 日志、metric、probe 与因果归因 | CI-0010、CI-0017～0025、CI-0029～0035、CI-0085、CI-0090～0092、CI-0103～0106、CI-0112～0117、CI-0124、CI-0184、CI-0189、CI-0192、CI-0201、CI-0244～0249、CI-0286、CI-0292、CI-0301、CI-0303、CI-0305、CI-0308～0309 | 先确认指标的 owner、坐标和采样支持 |
| 审查方法、文档、交付与信任 | CI-0043、CI-0081、CI-0124～0126、CI-0150～0151、CI-0171、CI-0186、CI-0211～0213、CI-0260～0261、CI-0269、CI-0274～0276、CI-0297、CI-0300、CI-0306、CI-0323、CI-0372、CI-0382～0386、CI-0399、CI-0417、CI-0424～0431 | 记录方法变化，不把情绪当技术证据 |

### 关键判断反转

| 最初判断或倾向 | 后来证据与修正 | 主卡 |
|---|---|---|
| 默认 `transition_detach` 能净化所有权 | 它会切断合法因果梯度，只能作为归因工具或局部隔离 | CI-0003 |
| `8×4` 崩溃或许说明四候选太少 | `8` 是 batch、`4` 是 candidate；崩溃来自多一个 singleton，结构问题是候选身份不稳定 | CI-0011、CI-0015、CI-0429 |
| 22/22 测试后 execution 已闭环 | 完整 epoch 仍显示 soft/hard dwell 与策略分布不同 | CI-0015、CI-0019、CI-0027、CI-0035 |
| V95 的 `flow_mag≈0.017` 已证明零流捷径 | 同单位数据探针表明短程真实运动稀疏；强指控撤回，但 V98 后来用 matched intervention 另行证明其地址无 action utility | CI-0063、CI-0100、CI-0105、CI-0430 |
| 独立 Stage1 / 单个 `t+48` Stage token 是长程答案 | Stage1 被单阶段联合训练吸收；单个 long token 被多时域空间 evidence 取代 | CI-0050、CI-0059～0061、CI-0065 |
| Flow 应独占最终空间地址 | 最终地址应是保留 identity fallback 的联合 posterior，Flow 同时承担 local-address 与 world evidence 两种角色 | CI-0066、CI-0068、CI-0121 |
| “不允许第二 action bypass”意味着禁止向现有 bottleneck 写地址 | 这是契约误读；应禁止重复 decoder，而非禁止合法信息写入唯一主路 | CI-0147、CI-0148、CI-0149 |
| 爆炸的 detail 路径可能就是有用细节变强 | 后续定位为学习到的高增益状态转换；显著梯度受害者不等于信息价值或根因 | CI-0159、CI-0160、CI-0414 |
| V120 是应整体恢复的完美结构 | V120 是行为/健康锚点，仍有语言、G3、W 时间等债务；恢复单位应是验证过的 operator/ownership | CI-0198、CI-0307、CI-0399、CI-0420～0421 |
| Schema24 的 S 缺陷解释 epoch 7–8 反弹 | 结构缺陷真实，但时间关联不足；反弹主要表现为 tail/gripper generalization | CI-0316、CI-0319 |
| Schema35 的 55× dynamic/static 比值是本版新制造 | 原始比值可能早已存在；Schema35 改变其 owner、预算和 reverse Jacobian，才形成新故障 | CI-0381、CI-0384 |
| S temporal prior 是非法旁路，应删除 | goal/history 可提供合法 bounded prior；非法的是独立或占主导，不是 prior 的存在 | CI-0405 |
| Schema39 的 W 整体更公共 | 对象轴反而较健康，真正严重的是四区间时间公共化 | CI-0413 |
| V27 是所有问题的最初源头 | V121 target 冲突、V122 identity 压弱和独立 mainline 重写更早；V27 是恢复岛后的再次断裂 | CI-0415、CI-0418～0420 |
| B-spline 曾是 ClearVLA 历史方案 | 本任务中仅是记录模板示例，没有版本、源码、日志或实验 | CI-0431 |

### 已否定或禁止直接复用的模式

- 通过 gain、quota、强制非空、entropy target 或额外 loss 掩盖 consumer 不闭环；
- 用 global null/dustbin 拒绝 protected fact、完整 W 或 consequence；
- 让 semantic/geometry/status 等互补值共享 winner-take-all type softmax；
- 用固定 `/3`、`/sqrt(3)`、K-mean 或 interval-mean 删除合法 common 信息；
- 先平均掉 K/camera/interval/type，再用 `expand`、位置标签或 learned query 伪造轴；
- 把 reliability、allocation、existence、validity 和 loss support 混成一个 mask；
- 让 predictor 同时提出并验证自己的 address，或让 auxiliary head 监督另一个未被 P 消费的字段；
- 用低参数量、低显存、非零梯度、低 auxiliary loss 或好看的单点 RMSE 宣称架构健康；
- 从历史版本整提交 cherry-pick，或把历史 audit 文档中的计划句当成当前实现授权；
- 为了“纯洁”删除正向信息；目标是清楚的 ownership 和受控组合，而非路径越少越好。

### 冻结终点的未决问题

这些问题在 visible message 5493 时仍未闭环，且不能自动视为当前 checkout 的问题：

| 未决问题 | 历史证据入口 | 仍缺少什么 |
|---|---|---|
| V25 row-normalized matching 与 partial OT 在同输入下，哪一个更真实地保存 Teacher identity/temporal bandwidth | `CI-0418`、`CI-0422` | 同输入、同权重、同日志口径的 target/posterior 与 downstream intervention |
| W interval publicization 中 Teacher target、W producer 和 terminal readout 各自贡献多少 | `CI-0388`、`CI-0395`、`CI-0413`、`CI-0422` | 三个边界逐一 matched zero/shuffle，并保持其他边界不变 |
| 合法 S temporal prior 在最终 interval terminal 中应具有怎样的相对杠杆 | `CI-0405`、`CI-0413` | S prior 与 W likelihood 的独立 intervention 和 action/tail 后果 |
| normalized arm/trajectory 已接近锚点时，physical gripper timing/calibration 为何仍明显落后 | `CI-0402`、`CI-0413` | normalizer、event timing、decoded gripper 与 dedicated heads 的同批对齐 |
| 稀有 late observation/address gradient spike 的数据/算子触发源是什么 | `CI-0403`、`CI-0414` | 首个触发 batch 的可复现 replay、producer activation/Jacobian 与输入样本审计 |

维护本账本时，只有出现新的历史证据、明确反转或新的决策边界才新增卡片；同一结论在新版本
复现时优先补 `【关联】` 或更新前置地图。当前源码故障进入当前问题账本，前瞻方案进入其他
auxiliary 文档，二者都不应伪装成新的历史事实卡。

详细跨版本叙述见文件末尾的
`Detailed whole-ledger synthesis through visible message 5493`。

## Extraction source and cursor

```text
Codex thread: 019f7501-e0e1-70f0-9f7b-9635d99b6f9a
Thread title: 主执行者 (originally 审查 V94 日志与源码)
Raw rollout:
  C:\Users\ASUS\.codex\sessions\2026\07\18\
  rollout-2026-07-18T19-34-50-019f7501-e0e1-70f0-9f7b-9635d99b6f9a.jsonl

Processed visible-message cursor: 5493
Last processed substantive turn: 2026-08-28, exact-history check of whether B-spline had been a real ClearVLA proposal
Next extraction point: none; frozen extraction target reached
Frozen extraction target: visible item 5493 (last substantive turn before the 2026-08-29 continuation request)
```

The cursor counts non-empty visible `user` and `assistant` message items in the
raw rollout. Injected plugin lists and environment contexts are indexed but are
not treated as project insights. Historical source line numbers below are the
line numbers cited in the conversation at that time; the current checkout may
have moved.


## Full chronological archive

The complete chronological archive (43 segments, 431 CI cards) is intentionally
externalized from the working copy. It remains recoverable from Git history:

- commit: `3665e39141ec3c38b018f6bfb1989eb1781354aa`
- path at that commit: `docs/research/auxiliary/CONVERSATION_INSIGHT_LEDGER.md`
- Git blob: `4f428336ee52beca364132f8ed480411da0b46c0`
- original file SHA-256: `39D37B0424F77800307FDE0CBE5BE451ED7124084F6C4B1DD0CBCECA685C75E0`

Retrieve or search it without restoring it to the working tree:

```powershell
git show 3665e39:docs/research/auxiliary/CONVERSATION_INSIGHT_LEDGER.md | rg 'CI-0413|Schema39'
```

This compact file is an index plus frozen synthesis. It is not current
architecture authority; re-check current source, run context and logs before
using any historical claim.

## Detailed whole-ledger synthesis through visible message 5493

### The durable result is an evidence discipline, not one winning version

Across 43 segments and 431 cards, the same distinction repeatedly separated a
real repair from a persuasive-looking half-fix:

```text
named module             != realized behavior
valid shape              != preserved axis identity
nonzero gradient         != useful downstream ownership
falling local loss        != task contribution
bounded forward value    != bounded backward Jacobian
source-correct model      != production lifecycle closure
low aggregate RMSE       != healthy tail/gripper/event behavior
immediate-parent gain    != recovery against historical anchors
```

The strongest reusable method is therefore end-to-end and bidirectional. Before
editing, trace every producer, transformation, consumer, loss, optimizer owner,
checkpoint/runtime call and diagnostic; after editing, trace the realized value
forward and ordinary task gradient backward. At each boundary preserve axis
identity, zero semantics, dtype, units, scale, repetition frequency and the
counterfactual showing that the consumer actually needs the information.

### Architecture history in six phases

1. **V93/V94 — execution semantics.** The project learned that a controller can
   be connected, trainable and still functionally dead; warmup can hide a dormant
   branch; and soft training, hard state advancement and deployment can describe
   different policies. Fixed candidate identity and complete lifecycle tests
   were more important than candidate count itself.

2. **V95-V100 — structured visual future evidence.** A pooled pseudo-top became
   Flow-DINO/JEPA with multiple horizons, then progressively more explicit raw
   detail and address ownership. The near-zero-flow episode established the
   canonical correction loop: inspect units/support, probe the data prior, then
   run learned/zero/shuffled interventions before forcing an internal magnitude.
   V96 removed a one-vector long-horizon bottleneck; V97/V98 showed that role
   names and raw-path gradients still did not prove the intended physical read.

3. **V101-V120 — causal localization and mature top-to-bottom behavior.** The
   work separated world phase, spatial address, factual detail and final action
   use; introduced progressive grounding, typed evidence, stateless intent and
   increasingly explicit future consequences; and repeatedly found that a
   supervised sidecar could remain optional at the policy. V120 retained real
   debt—weak language reads, small G3 correction and temporal publicization—but
   its main path remained behaviorally effective.

4. **V121/V122 and independent mainline — overcorrection and redesign.** V121
   introduced an exact Teacher target contradiction while also changing several
   adjacent owners. V122 further removed identity/common bandwidth. The
   independent mainline then reimplemented the whole peripheral graph rather
   than mechanically extracting it; G/W object identity collapsed almost from
   initialization. This is the clearest ancestry for later recovery work.

5. **Schema20-26 — behavior reconstruction and a recovery island.** Complete
   V120 source materialization exposed that modules, adapters, call order,
   optimizer geometry and lifecycle all belonged to the behavior contract.
   Schema24-fix and Schema25/26 recovered much of that behavior; V25/V26 became
   performance anchors, not architectural ideals. Their remaining debts included
   temporal W publicization, type competition, duplicate P3 carriers and
   static/dynamic precision imbalance.

6. **Schema27-39 — repeated ownership repairs and terminal-axis closure.** V27
   serially removed temporal identity, made W globally rejectable and deleted
   common precision. V31 independently reduced Teacher association bandwidth;
   V35 exposed unstable precision ownership. Later schemas restored many real
   axes and consumers. Schema39 finally put K/camera spatial termination in P2,
   interval termination in P3, protected dynamic precision once and preserved
   producer-owned FP32 probability measures. Its live evidence still showed
   weak temporal bandwidth and gripper calibration: a correct consumer topology
   did not repair a weak producer chain.

### Recurrent defect grammar

Most historical failures fall into eight reusable classes:

1. **Identity/provenance failure:** a version name, wrapper, checkpoint or log did
   not identify the graph that actually ran.
2. **Dormant lifecycle failure:** tests exercised warmup/CPU/train but not the
   activation, BF16, validation, endpoint, deployment or resume branch.
3. **Axis loss disguised by shape:** K, camera, interval, type, basis or lane was
   averaged away and later recreated by broadcast or a learned label.
4. **Optionality overreach:** null/dustbin/validity intended for one optional
   innovation gained authority to reject protected fact or the entire future
   path.
5. **Owner/consumer mismatch:** one tensor received the auxiliary target while a
   different hidden/public carrier reached action.
6. **Scale and numerical ownership failure:** fixed averaging, normalization of
   near-zero values, BF16 probability reconstruction, saturation or global
   clipping changed the effective learning path despite finite forward values.
7. **Diagnostic-semantic failure:** aggregate metrics, aliases, filtered probes
   or different owner names made a healthy-looking statistic describe the wrong
   field.
8. **Overcorrection:** a legitimate shortcut was removed by deleting surrounding
   legal information, often across several adjacent modules in one large change.

The common root is not insufficient cleanliness. It is failure to conserve
information and causal ownership across a continuous graph. `Pure` ownership
became dangerous when interpreted as exclusivity; the more durable goal is
clear, unique responsibility with every complementary fact still available to
its lawful consumer.

### Historical anchor roles at the frozen endpoint

- **V120**: regression, behavior and older-main-path health anchor; explicitly
  not a structurally perfect design.
- **V25/V26**: strongest available performance/replay island after the mainline
  reconstruction; not an unconditional rollback target, partly because full
  normalizer provenance was incomplete.
- **Immediate parent version**: only an attribution baseline for what the child
  newly changed, never a replacement for the two historical anchors.
- **No version**: accepted as a complete structural-health ideal. Recoverable
  units were producer/consumer semantics and verified operators, not whole
  commits.

### Historically recoverable semantics and rejected carryovers

The replay supported preserving independent complementary semantic/geometry
selection, real camera geometry, a unique protected consequence, static/dynamic
P1 ownership, consequence-innovation temporal operands, independent optional
lane nulls, exact Teacher target identities, progressive G, query-first local P1
and producer-owned FP32 measures. These are historical evidence boundaries, not
instructions to alter the current checkout.

It rejected or sharply limited type softmax over complementary evidence, fake
camera expansion, K/common centering that deletes valid detail, global W null,
dustbin-as-disappearance, reliability-as-loss-support, duplicate factual/effect
lanes, shared lane×basis null competition, fixed `/3` or `/sqrt(3)` fusion,
unbounded learned null, gain/quota/entropy patches and whole-version rollback.

### Unresolved at the frozen endpoint

The last evidence did not close five questions:

- whether V25-style row-normalized matching or partial OT better preserves true
  Teacher identity/temporal bandwidth under the same inputs;
- why W interval output remained much more public than its object axis and how
  much of that originated in Teacher targets versus W readout;
- how much lawful S temporal prior should influence the final interval terminal
  relative to action without becoming either dominant or normalized away;
- why normalized arm/trajectory quality could approach the performance anchors
  while physical gripper timing/calibration remained substantially worse;
- what data/operator event caused the rare late observation/address gradient
  spikes, beyond identifying their first visible dominant parameters.

Those are unresolved historical questions, not current blockers by declaration.
Current work must reverify them against the active source and current logs.

### Process correction and interaction history

Repeated frustration and loss of trust are recorded only where they changed the
technical standard: source snapshots replaced memory, pure issue ledgers replaced
success narratives, independent reviews covered runtime as well as the model,
and the handoff separated facts from hypotheses. Later apologies and calmer
reassessment likewise prevented anger-driven overcorrections such as discarding
the entire refactor or declaring dozens of versions uniformly invalid.

The completed extraction addresses the memory half of that failure. It does not
make historical conversation authoritative. For every new implementation the
active architecture contract, current source, serialized run context and current
evidence remain primary; this ledger explains ancestry, reversals and failure
patterns so they do not have to be rediscovered—or accidentally repeated.
