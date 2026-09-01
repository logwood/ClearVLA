# ClearVLA conversation insight ledger

Status: curated historical research memory, completed through frozen visible
message 5493, with a chronological evidence appendix; not an active architecture
contract and not implementation authority.

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

Active architecture truth remains in `../00_CURRENT_ARCHITECTURE_CONTRACT.md`.
Current blockers remain in `../CURRENT_MAINLINE_ISSUES.md`. The source-level
V120-to-Schema39 reconstruction remains in `ARCHITECTURE_REPLAY_LEDGER.md`.

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
through Schema39 claims should also be cross-checked against
`ARCHITECTURE_REPLAY_LEDGER.md` before reuse.

## Curated research map

这部分是账本的主入口。后面的 431 张卡是按时间保存的证据档案，不应再从头线性阅读，
也不应把每张卡都当作一个仍需修复的当前问题。

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

## Chronological evidence archive

以下 Segment 按对话发生顺序保留“当时为何这样判断、后来如何修正”的完整证据。它是
研究档案，不是待办列表，也不是当前架构说明。若同一主题在多个 Segment 重现，应以前面的
规范化主结论为入口，再用后续卡片补充版本证据，而不是把每次重现解释成一个新机制。

## Segment 001 — V93/V94 execution semantics and evidence discipline

### Segment boundary

```text
Date:                 2026-07-18
Visible messages:     2-79
Starts with:          clean-room static review of the current V94 source/log
Ends with:            interpretation of the old V92/V94 physical RMSE
Primary historical artifacts:
  scripts/current_v94_latent_ownership_execution.sh
  scripts/current_v93_native_dynamic_execution.sh
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/refinement.py
  clearvla/policy/trunk.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  attached old V93/V94/V92 nohup logs
```

### CI-0001 — Review the realized graph, not the version story

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The user requested a clean-room V94 audit based only on the current workspace
and the newest available V94 evidence, explicitly rejecting inherited conclusions
and code changes before diagnosis.

【当时的 insight / hypothesis】

An experiment identity is not established by its filename or banner. It is the
composition of the active wrapper chain, final CLI values, serialized policy and
trainer configs, active consumer code and the actual log. Missing evidence must
remain a stated limitation rather than being filled with an older version's
result.

【具体做法】

- Separated code facts from experimental judgments.
- Traced V94 through its wrappers into the active Evidence decoder.
- Compared each claimed switch with the field actually read by the active
  consumer.
- Refused to treat V91/V92 logs as V94 results when no V94 log was initially
  present.

【观察到的结果】

- The workspace initially had a V94 script but no V94 training log.
- V93 printed `[v92]` because it inherited the V92 wrapper, while its serialized
  output directory and `dynamic_block_route=1` identified the actual run.
- Later old logs also lacked a serialized git SHA, so an args-only comparison
  could not uniquely attribute behavior.

【当时的解释】

Several apparently model-level failures were first experiment-identity and
configuration-closure failures. A version label could not prove which path ran.

【后来怎么看（截至冻结终点）】

Retained. This became a recurring project rule: verify active script, run context,
source and log together. Later mainline/schema audits repeatedly confirmed the
rule and extended it to normalizer, source fingerprint and startup registry.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 2-5, 18-24, 38-43 and 73-76。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Experiment identity, source fingerprints, checkpoint provenance and all later
cross-version audits.

### CI-0002 — A declared switch is not a functional contract

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

V94 declared layer-gradient control, transition detach, z probes, learned dwell
and value supervision, but it was unclear whether the active Evidence path and
trainer consumed those values.

【当时的 insight / hypothesis】

A switch is functional only when its complete path closes:

```text
declaration -> wrapper precedence -> parsed config -> serialized owner
-> active consumer -> loss/runtime branch -> diagnostic evidence
```

Interface names, comments and non-null tensors do not establish closure.

【具体做法】

- Distinguished `latent_cvae_layer_grad_scale` from the active
  `layer_contract_grad_scale`.
- Checked whether the active Evidence trunk, rather than only legacy CVAE,
  consumed `latent_cvae_transition_detach`.
- Checked whether the z intervention existed in the active organizer.
- Traced `dwell_mode` into the trainer object read by `flow_losses()`.
- Added a preflight to inspect final arguments rather than wrapper intent.

【观察到的结果】

- V94's historical script declared `0.15`, while the active path still received
  a different layer-contract value.
- The original z probe and transition detach existed only in legacy CVAE.
- V93 serialized `dwell_mode=learned` in policy config, but the trainer lacked the
  field and defaulted to `fixed`; `evval` and value-reader gradient were exactly
  zero.
- Once the trainer boundary was repaired, early V94 showed non-zero value target
  spread, predicted spread and value-reader gradient.
- A later runtime check found an old wrapper could override the newly selected
  transition-detach default, demonstrating why final precedence matters.

【当时的解释】

The model did not fail to learn those functions; some functions were never on
the active path. Logging a requested value was insufficient evidence.

【后来怎么看（截至冻结终点）】

Retained as a general closure rule. By the frozen endpoint, the particular V94
wiring had been absorbed or replaced by later architectures and was not treated
as a donor implementation.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；具体 V94 接线已被后续架构吸收或替代。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预
- 对话：visible messages 5, 8-10, 18-24, 68-72 and 73-76。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`latent_cvae_layer_grad_scale`、`layer_contract_grad_scale`、`latent_cvae_transition_detach`、`dwell_mode`、`flow_losses()`。
- 补充直接证据：Historical V93 evidence: `evval=0`, value-reader gradient `0`。
- 补充直接证据：Historical repaired V94 evidence: non-zero value loss/spread/gradient by batches 20-420。

【关联】

Optimizer ownership, diagnostics truthfulness, serialized manifests and later
producer-to-consumer audits.

### CI-0003 — `transition_detach` changed from attribution tool to rejected default boundary

【时间 / 版本】

Legacy CVAE introduction on 2026-07-13; active V94 discussion and reversal on
2026-07-18.

【当时在解决什么问题】

The active Evidence decoder consumed `controlled_delta`, `rollout_effect_pred`
and `event_context`. The question was whether final action loss should update the
upstream dynamics that produced those transition features.

【当时的 insight / hypothesis】

`transition_detach` changes gradient ownership without changing the forward
value:

```text
detach=1: decoder reads transition evidence, but action loss stops at the boundary
detach=0: action loss also trains the upstream transition/dynamics producers
```

It is neither feature removal nor a decoder freeze.

【具体做法】

- Identified the first historical configuration/legacy use in commit `7f1976c`
  on 2026-07-13.
- Connected the switch to the active Evidence path on 2026-07-18.
- Initially proposed `detach=1` for an ownership experiment plus a `1 vs 0` A/B.
- After inspecting the other gradient discontinuities, changed the default to
  `0` and retained detach only as an explicit compatibility/causality boundary.

【观察到的结果】

- Before the active-path repair, the V94 comment claimed detach while the active
  graph remained attached.
- Making detach real plausibly reduced retained backward activations and GPU
  memory, while preserving forward values.
- The user then rejected a blanket ownership cut: legal upstream information
  should flow forward and receive ordinary action gradients backward.

【当时的解释】

The initial ownership experiment optimized attribution clarity, but it also
prevented the final task loss from correcting weak upstream evidence. Combined
with hard route selection and no-grad candidate probes, it made an already
fragmented gradient graph worse.

【后来怎么看（截至冻结终点）】

The same segment superseded the blanket-detach recommendation. The durable rule
is selective isolation: retain stop-gradient only for target leakage, teacher
isolation or an explicitly scoped causal experiment; do not use one global
detach to manufacture ownership.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留：显式梯度所有权概念保留；默认整块 detach 已否定。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 11-17, 27-28 and 54-66。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`transition_detach`、`controlled_delta`、`rollout_effect_pred`、`event_context`、`7f1976c`。

【关联】

Teacher isolation, lawful end-to-end gradients, causal information boundaries
and later complaints about disconnected top-to-action paths.

### CI-0004 — Host depth, operator aperture and compute cost are different quantities

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The logs showed `depth=3`, `effective_depth=8` and an execution cost near `0.011`.
These were being conflated with each other and with actual GPU work.

【当时的 insight / hypothesis】

Three axes must remain separate:

```text
host depth:        number of full MMDiT blocks/iterations
operator aperture: retained low-rank update directions inside an operation
execution cost:    either a declared proxy or measured wall-clock/FLOPs/memory
```

【具体做法】

- Reconstructed the logger field definitions for `evexec` and `evcap`.
- Derived the effective-depth formula from `rank=32`, `groups=4` and `ceil`.
- Traced the cost proxy to `progress * mean(capacity * dwell)`.
- Listed costs omitted by the proxy.

【观察到的结果】

- `effective_depth=8` came from one retained group of width eight; it was not an
  eighth MMDiT layer or one quarter of real compute.
- `depth=3` still meant three host operations.
- The cost proxy excluded fixed host attention/FFN, candidate probes, kernel
  timings, data movement, memory and actual FLOPs.
- Its loss weight was zero, so it was audit-only.

【当时的解释】

The historical logger named a quantized operator-capacity statistic as depth and
a capacity-dwell proxy as cost. Those names encouraged stronger conclusions than
the implementation supported.

【后来怎么看（截至冻结终点）】

Retained. A representation-control metric must not be reported as a compute
measurement. Real compute claims require wall-clock, peak memory and/or FLOP
instrumentation that covers all operations.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 29-40。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Later runtime/memory audits and the distinction between semantic capacity and
hardware efficiency.

### CI-0005 — `groups=4` hid small, potentially healthy capacity changes

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

Capacity appeared to jump between full rank 32 and effective rank 8. The user
argued that a modest reduction such as `32 -> 29` could be a valid result.

【当时的 insight / hypothesis】

Capacity control should be fine enough to represent small changes, and a model
should not be forced to display dramatic sparsity to prove that the controller
works. Control resolution and actual compute reduction are separate problems.

【具体做法】

- Identified `groups=4` as a convenience/stability heuristic without an
  experiment-backed reason that four was required.
- Proposed continuous ordered transparency during training or one-rank groups.
- Retained hard dispatch only as an evaluation/deployment option.
- Refused to add a cost loss merely to make the rank visibly decrease.

【观察到的结果】

- The hard forward could represent only `0/8/16/24/32`.
- A learned capacity of `0.92` still logged 32 because
  `ceil(0.92 * 4) = 4`.
- Historical V93 reached capacity near `0.01` and quantized effective depth near
  eight; historical V94 stayed near full capacity and depth 32.
- Changing group count alone would not avoid full-rank projection/reconstruction.

【当时的解释】

The quantizer both hid small control changes and made a one-group floor look like
a meaningful learned depth. Dramatic collapse could be a scale shortcut rather
than intelligent compute allocation.

【后来怎么看（截至冻结终点）】

The general principle is retained. The segment implemented `groups=32` plus
continuous transparency, but later architectures retired this controller and
the archive contains no matched experiment isolating that realization. It is a
scale/diagnostic lesson, not a donor operator.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；尺度原则保留，V94 的具体 realization 已被替代且未被独立归因。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 34-37 and 44-53。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0/8/16/24/32`。

【关联】

Fine-grained rank control, scale shortcuts, one-sided contracts and later
information-bandwidth discussions.

### CI-0006 — V93's execution controller was a post-block update controller

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The V93 mainline was advertised as learned dynamic execution, but the user
suspected that its control point was structurally misplaced.

【当时的 insight / hypothesis】

If a controller is intended to control execution, its decision must precede the
expensive operation. Applying capacity only to the produced residual controls
the update representation, not whether attention/FFN ran.

【具体做法】

- Expanded the actual wrapper and call chain.
- Traced controller, hard candidate selection, host execution, contraction and
  no-grad probes in their true order.
- Distinguished route, capacity, dwell and exit as separate decisions.

【观察到的结果】

- The dynamic path still ran a fixed three-iteration loop.
- It had current/next and dwell choices but no true skip/stop/no-op.
- The full host block ran before contraction.
- Candidate probes could rerun complete host operations but were omitted from
  the cost proxy.
- Historical route statistics showed little or no next-block usage.

【当时的解释】

The realized mechanism was better described as:

```text
fixed host operations + dynamic block/repeat choice + post-block direction aperture
```

It was not yet compute-aware scheduling.

【后来怎么看（截至冻结终点）】

Partially retained. The early repair moved the gate to the residual-writer
boundary, which improved semantic/gradient ordering but still did not skip QKV
or FFN FLOPs. The user explicitly deprioritized compute savings relative to a
correct action-gradient path.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；计算优化当时延期，控制点语义保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 34-40, 44-53 and 64-66。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Runtime lifecycle, static/dynamic recomputation and later performance audits.

### CI-0007 — Hard routing separated task learning from route learning

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The value reader had a dedicated candidate loss, but the chosen integer route
and dwell did not receive ordinary final-action gradients.

【当时的 insight / hypothesis】

An `argmin -> integer index -> selected operation` boundary is piecewise constant.
Non-zero controller/value gradients elsewhere do not prove that task loss can
train the discrete selection. Training and deployment may legitimately use
different routing algebra:

```text
train:  soft or straight-through candidate mixture
deploy: hard committed route
```

【具体做法】

- Distinguished capacity-mask gradients from route/dwell-selection gradients.
- Identified the dedicated value target as an indirect path rather than final
  action-loss closure.
- Implemented an attached candidate action chart and softmax mixture for
  training while retaining hard deployment dispatch.

【观察到的结果】

- Unselected candidates were outside the task-loss graph.
- V93's value reader received no dedicated supervision due to the trainer bug.
- Repaired V94 value-reader gradients proved its auxiliary loss was active, not
  that the hard selection boundary was differentiable.

【当时的解释】

The controller could learn candidate scores without learning how small score
changes affect the final action. This made route semantics dependent on an
auxiliary estimator and default/tie behavior.

【后来怎么看（截至冻结终点）】

The gradient-closure principle is retained. Later architecture lines did not
adopt this exact candidate-mixture controller as a recovery unit; its extra
operation paths and memory remain reasons not to recover it independently.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；梯度闭环原则保留，具体 candidate-mixture controller 已被替代。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 52-66。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Soft training/hard inference, action-path closure, optional route shortcuts and
later per-lane/null routing debates.

### CI-0008 — Cost was deliberately audit-only; capacity collapse was not a cost optimum

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

Historical V93 capacity rapidly approached a small value. It was unclear whether
the execution-cost objective caused this and whether a compute loss should be
used to control the mechanism.

【当时的 insight / hypothesis】

At this stage, action quality and gradient closure should determine useful
capacity. Compute cost should be observed but should not force compression. A
rank staying at 32 is not automatically a failure, and a drop to eight is not
automatically a success.

【具体做法】

- Verified `execution_cost_weight=0.0` and detached cost logging.
- Kept the proxy out of both the primary loss and candidate value target.
- Added separate capacity, soft/hard depth, route, dwell and real-runtime
  diagnostics as the desired accounting surface.

【观察到的结果】

- V93 capacity collapsed despite no active cost loss.
- V94 capacity stayed full while learned dwell quickly reached its maximum.
- Historical low capacity therefore behaved more like a learned residual-scale
  or aperture shortcut than a proven compute-quality tradeoff.

【当时的解释】

Without a true pre-operation compute boundary, adding a cost penalty would have
optimized a misleading proxy and risked sacrificing action quality. It would not
have made the fixed host work disappear.

【后来怎么看（截至冻结终点）】

Retained for this experimental phase. A future compute objective would require
a measured and causally complete cost surface, not merely a non-zero weight.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 21-37 and 44-66。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Execution-cost semantics, audit-only metrics, scale shortcuts and hardware
profiling.

### CI-0009 — Old V93 and old V94 failed in opposite execution modes

【时间 / 版本】

Old V93/V94 logs reviewed on 2026-07-18.

【当时在解决什么问题】

The old logs needed to be classified without confusing them with the newly
modified V94 source.

【当时的 insight / hypothesis】

Controller health is multi-dimensional. Non-zero gradients or a decreasing task
loss cannot establish useful route, dwell and capacity behavior. Each decision
surface must be observed separately.

【具体做法】

Compared progress, capacity, effective depth, dwell, route fractions, value
spread/common-mode, owner gradients and the available validation record.

【观察到的结果】

- Old V93/V92-style run: value reader remained dead, dwell stayed at one,
  capacity collapsed toward `0.01`, effective depth quantized near eight and
  next-block route was effectively unused.
- Old V94: value reader became active, but capacity stayed near one, effective
  depth stayed 32, dwell moved toward two and the value output contained a large
  common mode.
- Both used `transition_detach=1`, `rank=32`, `groups=4` and predated the later
  continuous/attached changes.
- The V93 log stopped before epoch-end validation, so it could not establish
  final effectiveness.

【当时的解释】

The earlier run used capacity as a cheap shrinkage control without a trained
value reader. The later run trained the value reader but mostly learned maximum
dwell/full capacity rather than discriminative routing.

【后来怎么看（截至冻结终点）】

Historical evidence only. These logs define failure modes and diagnostic needs,
not a donor architecture.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收为失败模式与日志口径。

【证据与锚点】

- 类型：日志观测；方案/设计记录；推断/未决
- 对话：visible messages 18-24, 41-49 and 73-76。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。

【关联】

Later controller diagnostics, common-mode collapse, route identifiability and
the rule that complete validation matters.

### CI-0010 — Low aggregate RMSE can coexist with poor action behavior

【时间 / 版本】

Old V92 validation reviewed on 2026-07-18.

【当时在解决什么问题】

The old run reported a seemingly low physical `full_rmse` near `0.098`, raising
the question of whether the policy was already strong.

【当时的 insight / hypothesis】

Aggregate raw-action RMSE is a dimension- and time-weighted average. It can be
made visually small by many arm dimensions, easy early timesteps and conservative
near-zero predictions while hiding gripper events and long-horizon failure.

【具体做法】

- Traced validation to decoded raw physical actions rather than normalized-space
  loss.
- Reconstructed full RMSE from six arm dimensions and one gripper dimension.
- Compared first/first4/first8/tail RMSE and event metrics.
- Checked teacher-forcing and target-action-leak indicators.

【观察到的结果】

```text
full_rmse          0.09818
arm_full_rmse      0.07841
gripper_full_rmse  0.17490
first_rmse         0.0327
first4_rmse        0.0411
first8_rmse        0.0533
tail_rmse          0.1142
tail/first ratio   3.49
gripper F1         about 0.381
event-head F1      about 0.197
pred/target events 2430 / 1357
deploy eligible    0
```

The six-to-one dimension weighting reproduces the reported full RMSE. Evaluation
did not use target action or teacher forcing, and proposal MSE gain was tiny.

【当时的解释】

The policy fit average arm motion and easy early steps, while gripper decisions,
event counts and the tail remained weak. Strong contraction could additionally
encourage smooth conservative actions.

【后来怎么看（截至冻结终点）】

Retained. Aggregate RMSE must always be paired with arm/gripper, first/tail,
horizon bands, event/motion and deployment metrics. Training flow loss and raw
validation RMSE are different quantities and must not be directly compared.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 73-79。
- 段级工件：见 Segment 001 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`full_rmse`。

【关联】

All later validation comparisons, normalized-versus-physical metric semantics,
conservative gripper behavior and early-platform diagnoses.

## Segment 001 synthesis

The earliest conversation already established the project's most durable audit
logic:

```text
version label is not run identity
declared switch is not a closed path
forward information and backward ownership must be audited separately
representation capacity is not hardware cost
non-zero gradient is not functional selection closure
aggregate RMSE is not policy health
```

It also contains an early warning that later became central: a locally clean
ownership mechanism can damage the global learning path when it removes legal
conditioning or gradients. The within-segment reversal from blanket transition
detach to selective causal isolation should therefore be preserved as an actual
insight, not edited out as a historical mistake.

## Segment 002 — Unified execution graph, loss accounting and diagnostic truth

### Segment boundary

```text
Date:                 2026-07-19
Visible messages:     81-166
Starts with:          first post-warmup crash in the new V94 soft candidate path
Ends with:            loss/logger refactor and confirmation that execution cost
                      remains a computed but non-optimizing audit statistic
Primary historical artifacts:
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/controller.py
  clearvla/policy/system.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  tests/test_time_domain_mmdit.py
  tests/test_v94_loss_logging.py
  complete old V93 log and successive V94 logs
```

### CI-0011 — Every scheduled boundary must be exercised, not merely parsed

【时间 / 版本】

V94, 2026-07-19.

【当时在解决什么问题】

The revised soft execution path trained normally through batch 200 and then
crashed on the first step for which `execution_progress > 0`.

【当时的 insight / hypothesis】

A preflight or warmup-only smoke cannot validate a dormant branch. Every
scheduled lifecycle boundary needs an executable test immediately before, at
and after activation, including backward propagation. A traceback site should
also be searched for duplicated algebra in sibling paths.

【具体做法】

- Mapped step 200 to the last fixed-path batch and step 201 to the first dynamic
  call.
- Traced candidate actions `[B,C,H,D]=[8,4,24,512]` and probabilities `[B,C]`.
- Found weights expanded to `[B,C,1,1,1]` instead of `[B,C,1,1]`.
- Confirmed by right-aligned broadcast arithmetic that the extra singleton made
  candidate batch `8` align against probability candidate count `4`; this was
  not matrix multiplication, controller-token count or rank grouping.
- Found the same mistake in both dynamic-route and ordinary learned-dwell paths.
- Replaced both with one shape-checking candidate-mixture helper.
- Ran a real PyTorch forward/backward across the boundary when pytest was not
  available in the same environment.

【观察到的结果】

- The error appeared exactly when progress first became positive, not from OOM,
  data, NaN or divergence.
- The reported `8 vs 4` was batch size versus candidate count, not controller
  token count.
- After repair, the exact-log-shape forward/backward, dynamic axes, value-reader
  task gradient and eval committed-only checks passed.
- The existing relevant test had not actually been run successfully before the
  experiment.

【当时的解释】

The first 200 batches exercised only the old path. Their healthy loss said
nothing about the newly added mechanism. The duplicated broadcast error showed
that local traceback repair without sibling-path review would have left a second
latent failure.

【后来怎么看（截至冻结终点）】

Retained. Scheduled transitions, ODE endpoints, training/eval branches and
feature-flag activation boundaries must all be explicit test points.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；具体广播故障已修复。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 81-97 and retrospective clarification 5464-5465。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Later preflight failures, teacher-forced boundaries, five-step sampling and the
mandatory subsystem-familiarity rule.

### CI-0012 — A complete run can optimize the task while its named mechanism is dead

【时间 / 版本】

Complete V93 eight-epoch log reviewed on 2026-07-19.

【当时在解决什么问题】

The complete old V93 run reached a strong aggregate RMSE, raising the question
of whether dynamic execution had eventually learned despite weak early signals.

【当时的 insight / hypothesis】

Task improvement is not evidence that an auxiliary mechanism contributed. A
named route/controller must be evaluated through its own state transitions,
consumer effects and owner gradients across the entire run.

【具体做法】

- Read all eight epochs and 23,776 steps.
- Compared route-next fraction, dwell, value-reader gradient, per-block gradient,
  capacity/depth, candidate probes and validation/event trends.

【观察到的结果】

```text
full RMSE:                    0.0982 -> 0.0807
first RMSE:                   0.0327 -> 0.0092
tail RMSE:                    0.1142 -> 0.0947
tail/first:                   3.49 -> 10.25
dynamic next-route fraction:  0 for all epochs
dwell expected:               1 for all epochs
value-reader gradient:        0 for all epochs
block 1/2 gradient:           0 from epoch 2 onward
effective rank:               converged near one coarse group, about 8
```

Gripper/event behavior peaked earlier and then became increasingly conservative;
event-head F1 fell to about `0.0535` while aggregate RMSE kept improving.

【当时的解释】

Tie/default behavior selected block 0 and dwell 1. Those choices starved the
other blocks, reinforcing the collapse. Candidate probes consumed computation
but did not change the committed strategy. The remaining shallow path was still
powerful enough to fit average actions.

【后来怎么看（截至冻结终点）】

Retained as an archetypal false-positive experiment: successful optimization of
the base task with a nonfunctional claimed mechanism.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为失败模式；V93 本身不作为动态执行成功版本。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 98-103。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Mechanism JVP/intervention requirements, per-owner gradients and the later rule
that losses/gradients alone do not prove action consumption.

### CI-0013 — Connected gradients can still produce an uninformative ensemble

【时间 / 版本】

Early revised V94, batches 220-380, 2026-07-19.

【当时在解决什么问题】

The new candidate mixture gave the value reader and controller non-zero task
gradients, but it was unclear whether they learned a useful execution policy.

【当时的 insight / hypothesis】

`gradient != 0` proves connectivity, not discrimination or ownership. A soft
selector can settle into an almost uniform ensemble that weakens each candidate's
learning signal while the selector itself continues to receive gradients.

【具体做法】

- Compared selection entropy to `ln(4)` and maximum candidate probability.
- Compared predicted versus target candidate spread, exact decision accuracy
  and common-mode ratio.
- Aligned operation-block gradients with old V93 over the same batches.
- Separated soft policy behavior from hard committed audit behavior.

【观察到的结果】

- Value/controller gradients increased by roughly two orders of magnitude over
  the dead V93 path.
- Many batches were nearly `[0.25,0.25,0.25,0.25]` with entropy `1.386`.
- Predicted candidate spread covered only about 10-30% of target spread early.
- Hard dwell moved to two while the soft policy remained almost uniform.
- Average MMDiT block gradient fell by roughly 80%, with several exact-zero
  batches while selector gradients remained non-zero.
- Runtime became about 8.8% slower without an early task advantage.

【当时的解释】

The repair moved learning pressure into the selector, but the soft average
diluted candidate-specific operation gradients. Tiny score differences drove
hard audit choices before the soft policy had meaningful confidence.

【后来怎么看（截至冻结终点）】

Retained. Connectivity, variation, matched intervention and downstream action
effect are distinct requirements. The exact four-candidate implementation was
subsequently replaced within this same segment.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为诊断原则；四候选混合方案已被当时后续重构替代。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；方案/设计记录
- 对话：visible messages 104-116。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Later publicization/common-mode problems, uniform selectors and weak W/S action
consumption.

### CI-0014 — Progress must interpolate behavior, not merely a scalar inside a new graph

【时间 / 版本】

V94 warmup transition, 2026-07-19.

【当时在解决什么问题】

At step 200 the model ran `block0 -> block1 -> block2`; at step 201 it instantly
switched to a repeated current/next-by-dwell candidate graph, even though
`progress` was only about `0.001`.

【当时的 insight / hypothesis】

A schedule is continuous only if the realized behavior graph is continuous.
Interpolating capacity inside a newly activated topology does not soften the
topology change. Warmup and learned execution need the same candidate coordinate
system from the start, with control mass transferred gradually.

【具体做法】

- Compared the fixed and dynamic call graphs.
- Measured the step-200-to-201 output discontinuity on a small CPU experiment.
- Checked what candidate comparisons the value reader saw before and after the
  boundary.

【观察到的结果】

- Opening progress by only `0.1%` changed output by about `15.2%` of its RMS.
- Warmup trained only within-block dwell comparison; dynamic mode suddenly
  required cross-block calibration.
- Target spread remained while predicted spread and accuracy collapsed after
  the switch.

【当时的解释】

The schedule changed semantics rather than gradually enabling one semantics.
This distribution shift explained abrupt gradient redistribution, weak value
calibration and unstable hard choices better than “training is still early.”

【后来怎么看（截至冻结终点）】

Retained. Curriculum variables must interpolate the actual producer-consumer
path, tensor support and target semantics, not only multiply an output.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；当时的 fixed/dynamic topology split was removed.

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 110-127。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Later progress/phase discussions, training-deployment distribution shifts and
future-target interval semantics.

### CI-0015 — Training state, training output and deployment must share one coordinate system

【时间 / 版本】

V94 execution rewrite, 2026-07-19.

【当时在解决什么问题】

The old revision used a soft candidate mixture for the current training output,
a hard argmin to choose the next training state and a hard committed operation
for deployment/logging.

【当时的 insight / hypothesis】

Three semantically different execution graphs cannot jointly define one learned
controller. At small fixed depth, training state can remain differentiable and
monotonic; hard dispatch should be a deployment/audit projection of the same
candidate space, not a hidden recurrent state transition.

【具体做法】

- Unified fixed, warmup and learned phases on a `3 blocks x 2 dwell = 6`
  candidate table.
- Propagated a differentiable monotonic pointer during training.
- Removed hard block choice from next-step training state.
- Used a fixed decision clock and kept hard choices for deployment/audit.
- Made progress transfer mass between neutral and learned behavior rather than
  exchange the graph.

【观察到的结果】

- The reported step-200-to-201 change became `0.19%` of the fully enabled change.
- Candidate tensors remained on one axis at all schedule points.
- The later retrospective clarified that the change was not "four candidates
  are too few." It replaced four locally redefined rows with six globally fixed
  identities; the coordinate system, not the raw count, was the defect.
- The historical review reported 22/22 structural regressions passing.
- Later epoch validation still found a soft/hard dwell gap, showing that sharing
  candidate coordinates did not by itself make the two policies equivalent.

【当时的解释】

The rewrite closed the recurrent hard-state discontinuity while retaining a
deployable hard projection. However, a soft optimizer and a hard argmin can still
have different expected behavior even on the same coordinates.

【后来怎么看（截至冻结终点）】

Partially retained. The coordinate-system and no-hidden-hard-state principles
remain durable; the claim that the full execution structure was “finished” was
too strong because later logs exposed a remaining soft/hard policy gap.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；结构原则保留，完整闭环声明被后续证据修正。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 117-131 and 144-150; retrospective clarification 5466-5467。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Train/deploy sampling probes, endpoint heads and later five-step deployment
contracts.

### CI-0016 — Remove unidentifiable value gauges before tuning their scale

【时间 / 版本】

V94 value reader, 2026-07-19.

【当时在解决什么问题】

The value prediction placed nearly all energy in a candidate-common direction,
while softmax and the centered auxiliary target cared only about relative values.

【当时的 insight / hypothesis】

If both the loss and decision are invariant to a common offset, that offset is
an unidentifiable gauge. Giving it a free bias wastes capacity and can dominate
diagnostics. The reader should predict a zero-mean, scale-calibrated advantage,
not an arbitrary absolute value.

【具体做法】

- Removed/compatibly consumed the value-head bias.
- Centered candidate outputs exactly.
- Standardized physical candidate targets by their decision spread so that
  temperature one operated on a meaningful dimensionless scale.
- Added predicted/target spread, correlation, pair accuracy and common-mode
  diagnostics.

【观察到的结果】

- Common-mode ratio fell from about `0.998` to about `0.002` in the matched early
  window and remained low through epoch one.
- Predicted-spread coverage roughly doubled early.
- Later epoch-one evidence showed correlation about `0.544`, pairwise accuracy
  about `0.809`, but exact hard decision accuracy only about `0.484` and predicted
  spread still only about 53% of target.

【当时的解释】

Removing the gauge fixed a real structural waste, but did not make candidate
ranking perfect. Gauge removal and identifiability are prerequisites, not a
substitute for informative inputs or supervision.

【后来怎么看（截至冻结终点）】

Retained as a general algebra rule. Similar public/common free directions recur
later in S/W/P and must be treated by owner semantics rather than arbitrary gain.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 110-131 and 137-150。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Public/private carrier separation, zero-centered innovations and later common
world residual failures.

### CI-0017 — Audit metrics must be detached, correctly weighted and semantically named

【时间 / 版本】

V94 execution review, 2026-07-19.

【当时在解决什么问题】

Capacity, effective depth and execution cost could carry gradients or be biased
by terminated padding and equal-per-step aggregation, despite being interpreted
as audit statistics.

【当时的 insight / hypothesis】

An audit metric must not silently become an objective. Its aggregation measure
must match real executed mass, and padding/no-op states must not masquerade as
zero capacity. Metric names must reveal whether they report a soft policy, hard
policy, scheduled mixture or actual hardware work.

【具体做法】

- Detached scalar audit values and candidate teacher while retaining the
  trainable candidate value field.
- Weighted capacity/depth by actual operation mass rather than equally by
  decision slot.
- Excluded termination padding from effective-rank accounting.
- Split soft/hard route, soft/hard dwell, actual/learned entropy and capacity
  diagnostics.

【观察到的结果】

- Audit quantities no longer had autograd ownership.
- Empty terminal steps no longer reduced reported depth toward zero.
- A later comparison showed old and new cost numbers were not directly
  comparable because the old proxy was multiplied by progress while the revised
  audit represented scheduled soft execution.
- `execution_cost_weight` remained exactly zero.

【当时的解释】

Incorrect aggregation can create a convincing but false story of adaptive
capacity. Detaching and naming the audit surface protects the model from the
metric and protects analysis from hidden measures.

【后来怎么看（截至冻结终点）】

Retained. Diagnostics have an ownership contract just as model tensors do.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 121-131, 137-150 and 164-166。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`execution_cost_weight`。

【关联】

Runtime profiling, gradient sentinels, metric coverage and later logger audits.

### CI-0018 — Compare aligned training support and reconstruct the loss before interpreting it

【时间 / 版本】

Old/new V94 comparison, 2026-07-19.

【当时在解决什么问题】

The revised run initially had a slightly higher total loss, while the old run
had many more batches and a differently scaled value objective.

【当时的 insight / hypothesis】

Cross-version training comparisons must align batch/step, schedule progress,
data/seed and active loss semantics. Total loss cannot be interpreted until it
is reconstructed from raw values, effective weights and contributions.

【具体做法】

- Compared only the shared batch 20-420 range.
- Subtracted the newly standardized value contribution before comparing the
  core task.
- Verified the progress-zero boundary by matching component values.
- Continued comparisons at aligned 640-820 batches and then full epoch one.

【观察到的结果】

- At batch 20, almost the entire total-loss difference equaled
  `0.05 * new_value_loss`; pflow/decode/first/tail were nearly identical.
- Over batches 220-420, core loss improved about 1.7% despite total loss being
  about 1.2% higher.
- Over 640-820, physical flow improved about 34%, but rollout improved only
  about 1%.
- At epoch one, physical flow and arm validation improved, while gripper-tail
  RMSE worsened and event-head generalization weakened.

【当时的解释】

The early higher scalar loss was bookkeeping, not worse task behavior. The later
uneven gains showed that one headline metric could still hide an unchanged
rollout objective or a regressing action subtype.

【后来怎么看（截至冻结终点）】

Retained. Aligned support and contribution reconstruction became the basis for
later recovery gates and eight-epoch comparisons.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 128-150。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Loss ledger, raw versus normalized metrics, version fingerprints and complete
epoch comparisons.

### CI-0019 — Soft training and hard evaluation require a paired causal check

【时间 / 版本】

V94 epoch-one validation, 2026-07-19.

【当时在解决什么问题】

After the unified candidate rewrite, the soft and hard policies shared a
coordinate system but had materially different expected dwell.

【当时的 insight / hypothesis】

Train/deploy mismatch is a structural risk, but correlation with a validation
failure does not prove causality. The highest-value test is paired evaluation of
the same checkpoint and samples under soft versus hard execution, without
retraining or adding surrogate gradients.

【具体做法】

- Compared soft/hard route and dwell statistics.
- Related the gap to gripper-tail and event-ratio behavior without claiming a
  unique cause.
- Proposed same-checkpoint soft/hard validation as the discriminator.

【观察到的结果】

- Soft/hard route fractions were close.
- Soft dwell was roughly `1.16-1.18`; hard dwell often `1.42-1.63`.
- Full and arm RMSE improved, while gripper full/tail worsened by about 3.4%/5.2%
  and tail/first slightly exceeded its deployment threshold.
- Hard exact candidate accuracy remained below 50% despite improved pairwise
  ranking.

【当时的解释】

Repeated hard operations could plausibly amplify tail/gripper behavior, but the
log alone could not establish that this was the only cause. A paired intervention
would separate routing mismatch from event-target or representation problems.

【后来怎么看（截至冻结终点）】

Retained as a causal-testing rule. The later complete V94 evidence confirmed the
substantive mismatch: soft optimization, hard state advancement and deployment
could select different effective policies even after local legality tests passed.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；后续 `CI-0027` 提供了该 soft/hard mismatch 的版本级证据。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；方案/设计记录
- 对话：visible messages 144-150。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Sampling-path probes, five-step deployment and train/deploy endpoint contracts.

### CI-0020 — Loss composition is an executable accounting contract

【时间 / 版本】

V94 loss audit, 2026-07-19.

【当时在解决什么问题】

The stable run made it possible to ask whether historical auxiliary losses were
duplicated, bypassed their configured weights or inherited an unrelated decay
schedule.

【当时的 insight / hypothesis】

Every active objective needs one semantic owner, one effective weight and one
visible contribution. A total loss should be exactly reconstructible; its
residual is an invariant that detects missing, repeated or overwritten terms.

【具体做法】

- Reconstructed epoch-one total loss to within `3.6e-7`.
- Traced rollout predictions/targets under `controlled_base_mode=fixed_zero`.
- Compared layer milestone-delta and rollout-delta implementations.
- Traced `layer_contract_loss_weight` through its producer and the policy-stage
  merge.
- Separated layer-contract scheduling from inherited midcut scheduling.

【观察到的结果】

- `rollout_effect_pred == rollout_delta_pred` and the corresponding targets were
  equal in fixed-zero mode, so two objectives duplicated the same supervision.
- Layer milestone-delta and layer rollout-delta were also identical.
- The policy merge consumed raw `layer_contract`, bypassing the producer's
  configured weighted `loss`; the weight interface was functionally ineffective.
- A structural layer contract inherited a schedule that decayed its external
  scale by about 85% over four epochs.
- The main physical flow still owned about 68% of the total loss; the issue was
  semantic duplication and control, not wholesale auxiliary domination.

【当时的解释】

These were history-accumulation defects: aliases and old scheduling surfaces had
become independently weighted owners of the same signal. A working optimizer
could not reveal that semantic duplication.

【后来怎么看（截至冻结终点）】

Retained. The segment removed the duplicate terms, made layer-contract weight
effective, gave it an independent constant scale and added grouped contributions
plus `ledger_residual`.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；列出的确定故障在当时已修复。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 151-163。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`layer_contract_loss_weight`、`layer_contract`、`ledger_residual`。

【关联】

Later loss ledgers, current-loss support, teacher masking and parameter-semantic
audits.

### CI-0021 — Logging must encode absence, zero and failure as different states

【时间 / 版本】

V94 logger audit, 2026-07-19.

【当时在解决什么问题】

The Evidence decoder fell through a legacy formatter containing hundreds of
fields and default-zero lookups. Active probes could also be computed and then
discarded by an upstream prefix whitelist.

【当时的 insight / hypothesis】

`missing`, `inactive`, `measured zero` and `failed to collect` are different
states. Console logging should expose active semantics compactly while complete
machine-readable scalars remain available for offline audit. Zero suppression
must be semantic, not numeric.

【具体做法】

- Counted a roughly 4.9 KB batch line, 311 named groups and 378 zero-valued
  numbers from the legacy fallback.
- Traced Evidence z-probe computation into a sampler whitelist that dropped the
  `evidence_*` prefix.
- Replaced the fallback with Evidence-specific train/exec/grad and epoch/val/probe
  lines.
- Kept complete metrics in JSONL.
- Hid inactive placeholder zeros but retained meaningful zero gradients,
  violations, ledger residuals and deployment flags.
- Added motion-head validation and soft/hard routing diagnostics.

【观察到的结果】

- The old `sample_latent_cvae_* = 0` fields belonged to an inactive path and
  could not diagnose the active Evidence latent.
- Evidence z-zero/z-shuffle was computed in eval but filtered before aggregation.
- Motion loss contributed about as much as event loss but previously had no
  validation precision/recall/F1.
- After the refactor, total loss could be audited through named groups and a
  near-zero residual without dumping every legacy field to nohup.

【当时的解释】

The logger was not merely ugly: it erased the distinction between unimplemented,
disabled, zero and filtered measurements, which made structural attribution
unreliable.

【后来怎么看（截至冻结终点）】

Retained. Logging is part of the experimental interface and must evolve with the
active graph, but compact output must not replace complete machine-readable
evidence.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；当时的 V94 formatter/probe gap已修复。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 144-166。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

The later ClearVLA log-audit skill, metric naming reforms and non-finite failure
records.

### CI-0022 — Auxiliary targets must be aligned with the failure they are meant to judge

【时间 / 版本】

V94 loss/validation audit, 2026-07-19.

【当时在解决什么问题】

Arm metrics improved substantially while gripper tail and event behavior lagged.
The controller value target, event weighting, motion objective and proposal
utility had incomplete or ambiguous semantics.

【当时的 insight / hypothesis】

Before changing an auxiliary weight, determine what decision it ranks, which
failure mode it observes and whether validation measures the same object. A
target matching average action dimensions may be mathematically consistent but
misaligned with deployment-critical rare events.

【具体做法】

- Identified value aggregation of roughly six arm dimensions to one gripper
  dimension.
- Requested arm, gripper and joint candidate regret plus selected-versus-oracle
  regret before reweighting.
- Found `event_positive_weight=4` meant total positive weight five in one path,
  while another path interpreted the parameter differently.
- Noted motion's material loss share without matching validation metrics.
- Kept the low-weight proposal loss pending a better paired-utility estimate
  rather than deleting it from sparse evidence.

【观察到的结果】

- Epoch-one arm RMSE improved strongly while gripper-tail RMSE regressed.
- Motion and event each contributed roughly 4.6% of total loss.
- Proposal utility was tiny and measured on only about 8.5% of validation
  batches.

【当时的解释】

These facts raised a plausible target-alignment problem but did not uniquely
prove causality. Immediate gripper reweighting could have treated the symptom or
distorted the primary physical objective.

【后来怎么看（截至冻结终点）】

Retained as an unresolved-assumption discipline: instrument matched outcomes
before changing behavior weights. The frozen archive does not link an isolated
result for this exact auxiliary target, so the method survives without a
version-level benefit claim.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；matched-outcome 方法保留，原 auxiliary target 收益未被隔离。

【证据与锚点】

- 类型：日志观测；测试/运行观测；历史检索；推断/未决
- 对话：visible messages 150-163。
- 段级工件：见 Segment 002 boundary 的 `Primary historical artifacts`。

【关联】

Rare-event supervision, balanced-score debates, current loss support and later
gripper conservatism.

## Segment 002 synthesis

This segment corrected an important overclaim from its own midpoint. It first
reported the execution rewrite as structurally complete, then full logging and
validation showed that soft optimization and hard evaluation still differed.
The durable lesson is therefore not the particular six-candidate controller; it
is the stronger review procedure:

```text
exercise every lifecycle boundary
trace one coordinate system across train state, train output and deployment
remove algebraically unidentifiable gauges
prove selection variation and consumer effect, not only gradient existence
align comparisons by data, step, schedule and loss semantics
make loss and logging executable accounting contracts
```

This is also the first segment in which logging itself became recognized as a
model-research subsystem rather than presentation code. A filtered probe or a
default zero can erase the only evidence capable of distinguishing a live owner
from a dead branch.

## Segment 003 — V94 false convergence, execution baselines and audit semantics

### Segment boundary

```text
Date:                 2026-07-19
Visible messages:     167-264
Starts with:          multi-epoch diagnosis of the old V94 run
Ends with:            metric-only interpretation of fix2 through batch 2020
Primary historical artifacts:
  scripts/current_v94_latent_ownership_execution.sh
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/controller.py
  clearvla/model/latent_cvae/refinement.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  clearvla/tools/audit_policy_logs.py
  .agents/skills/audit-clearvla-logs/
  attached V92/V93/V94 nohup logs and epoch JSONL records
```

### CI-0023 — Old V94 was a structural false convergence, not a numerical crash

【时间 / 版本】

Old V94 and its first ownership/execution repair, 2026-07-19.

【当时在解决什么问题】

An older V94 log showed rapidly falling training loss but weak validation,
nearly full capacity, poor long-horizon behavior and inconsistent gripper
events. The question was whether more epochs could repair it or whether the
active graph was structurally biased toward an easier path.

【当时的 insight / hypothesis】

A run can be finite, optimize quickly and still converge around the wrong
owners. Cold-started modules, detached or attenuated ownership gradients, a
saturated capacity initialization and an action-state shortcut can jointly make
the primary training loss look healthy while the intended execution mechanism
remains unidentifiable.

【具体做法】

- Compared epoch-level train flow, sampled validation RMSE, first/tail, capacity,
  value-reader discrimination, gradients and decoded event counts.
- Reconstructed the actual configuration rather than trusting the V94 label:
  `operator_groups=32`, capacity logit `4.0`,
  `latent_cvae_layer_detach=1`, two gradient scales of `0.15` and upper LR
  scale `0.2`.
- Noted that the Stage-1 load skipped roughly 303 dirty adapter keys and left
  572-573 missing keys, so much of the new execution path was a cold start.
- Separated the easy action-state path from layer ownership, execution value and
  long-horizon objectives.

【观察到的结果】

- Train physical flow fell from about `0.2740` to `0.0891` between epochs 1
  and 2, while validation RMSE improved only from `0.11576` to `0.11285`.
- Arm validation slightly worsened and tail/first rose from `2.26` to `2.36`.
- Capacity remained essentially `1.0`, effective basis count remained
  `32/32`, and capacity gradient decayed toward `5.7e-7`.
- Value decision accuracy stayed near `0.53` and its common-mode ratio reached
  approximately `1.0`.
- Decoded gripper events were about `4.29x` the target count even though the
  auxiliary event accuracy looked high.
- `rollout_dynamics_loss` and `rollout_delta_loss` were numerically
  duplicated in that configuration.

【当时的解释】

The log described a full-rank, weak-ownership attractor. The model could reduce
local flow through the already strong action path, while the randomly
initialized structural path received weaker gradients and the capacity control
had almost no behaviorally distinguishable alternative to learn.

【后来怎么看（截至冻结终点）】

Partly confirmed by aligned later runs. Removing detach/attenuation and restoring
the execution gradients reduced the value common mode from about `1.0` to
`0.01-0.02`, restored controller/basis gradients and improved epoch-two
validation RMSE to `0.09654`. It did not, however, prove that learned capacity
or execution improved over a neutral/full-capacity policy.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留：旧版根因已得到强证据；execution 的净任务收益当时仍未证明。

【证据与锚点】

- 类型：日志观测；因果干预；推断/未决
- 对话：visible messages 167-176, 206-219 and 235-239。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`rollout_dynamics_loss`、`rollout_delta_loss`。

【关联】

CI-0012, CI-0013, capacity saturation, cold-start ownership and later execution
ablation.

### CI-0024 — A reusable log audit must reconstruct lineage and evidence coverage

【时间 / 版本】

V94 audit tooling, 2026-07-19.

【当时在解决什么问题】

Manual reading was repeatedly mixing raw losses with weighted contributions,
counting multiple truncations of one run as independent evidence and treating a
log with no validation as if it had passed validation.

【当时的 insight / hypothesis】

A log parser is useful only if it understands experiment identity, objective
accounting and missing evidence. Regex extraction without run lineage and source
semantics can automate the same wrong conclusion faster.

【具体做法】

- Built the project-level `audit-clearvla-logs` skill and
  `clearvla.tools.audit_policy_logs`.
- Added support for old V39/V92/V93 rows, compact V94 rows, epoch JSON/JSONL and
  automatic merging of a run directory's nohup plus epoch ledger.
- Divided output into objective contribution, training/generalization,
  structure, gradient, controller, data and runtime views.
- Added a metric catalogue and source map so anomalous metrics could be traced
  back to the active producer.
- Deduplicated 15 attachments into about five independent run lineages using
  headers, output directories, configs and coverage instead of attachment count.
- Aligned comparable runs at the same batches and explicitly marked
  non-single-variable comparisons.

【观察到的结果】

- At batch 20, three lineages began with physical-flow values
  `1.377685 / 1.377680 / 1.377701`, supporting a highly comparable initial
  data/seed state despite later code differences.
- The tool correctly exposed duplicate objectives, capacity saturation, dead
  gradients, value common mode, event-rate mismatch and legacy placeholder
  pollution.
- The latest pre-validation run initially appeared as `0 warnings`; manual
  review showed this meant “validation absent,” not “validation healthy.”
- Legacy logs contained roughly 76-81% zero placeholders, whereas the compact
  path reduced that ratio to about 3%.

【当时的解释】

Evidence completeness is itself a typed state. `unmeasured`, `pending`,
`inactive`, `measured zero` and `passed` cannot share one fallback value.
Likewise, ten attachments are not ten votes when most are truncations of the same
run.

【后来怎么看（截至冻结终点）】

Retained, with an explicit limitation: the skill was judged useful but not a
replacement for source-backed interpretation. Automatic tail comparison still
needed same-step alignment and a clear “unassessed” state for missing
validation.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；工具已建立，证据覆盖语义仍需持续维护。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 180-194, 206-219 and 235-239。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`clearvla.tools.audit_policy_logs`。
- 直接量值/表达式（卡内原文）：`1.377685 / 1.377680 / 1.377701`。

【关联】

CI-0001, CI-0017, CI-0018, CI-0020 and CI-0021.

### CI-0025 — Metric names must carry phase, statistic and structural axis

【时间 / 版本】

V94 logging vocabulary audit, 2026-07-19.

【当时在解决什么问题】

Correctly computed values were still being misread because one short label was
used for different stages or axes: `depth`, `first8`, `tail`, `event`,
`cost`, `value` and several compressed abbreviations.

【当时的 insight / hypothesis】

A metric name is an interface. It should reveal enough of the object, lifecycle
stage and statistic to prevent an invalid comparison, without embedding an
entire source path in every console row.

【具体做法】

- Distinguished decoder block count, operator soft-equivalent basis mass and
  dwell depth instead of calling all three `depth`.
- Distinguished train flow MSE-like quantities from sampled validation action
  RMSE.
- Established `*_loss_raw`, `loss_contrib_*`, `loss_group_*`,
  `loss_total` and `loss_ledger_residual` as separate accounting layers.
- Distinguished decoded gripper events from auxiliary event-head
  classification, and made count ratios visibly non-probabilistic.
- Kept canonical JSON keys for compatibility while moderately expanding compact
  display names and teaching the parser both vocabularies.
- Retained common abbreviations such as `rmse`, `f1`, `grad`, `soft` and
  `hard`, while removing opaque labels such as `pfn`, `rstep`,
  `stdr` and `dnratio`.

【观察到的结果】

- Train `first8/tail/arm/grip` referred to flow-matching errors, while
  validation used sampled-action RMSE under the same display names.
- `effective_rank=31.8` described a continuous basis mass with all 32 channels
  still computed, not 31.8 executed Transformer layers or pruned rank.
- `event_ratio` could legitimately exceed one.
- `proposal_gain` was an MSE difference on limited ablation coverage, not a
  reward.
- `z_zero/z_shuffle` measured condition change, not action RMSE.
- New and old display-name parser tests passed while serialized fields remained
  backward compatible.

【当时的解释】

The most dangerous logger error was not a wrong number but a plausible-looking
number with the wrong unit or axis. In particular, a training flow value and a
validation RMSE cannot define a generalization ratio merely because both are
called `tail`.

【后来怎么看（截至冻结终点）】

Retained. Later in the same segment, `terminal_probability` was found to mix
new exit mass with already-terminal occupancy, showing that semantic naming must
also follow state accumulation, not only tensor shape.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；部分 V94 compact names were repaired, historical JSON aliases retained.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 195-205, 243-256 and 257-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`loss_total`、`loss_ledger_residual`、`first8/tail/arm/grip`、`event_ratio`、`proposal_gain`、`z_zero/z_shuffle`、`terminal_probability`。

【关联】

Metric catalogue, loss ledger, runtime timing and all later recovery audits.

### CI-0026 — An execution controller needs the committed state as a candidate

【时间 / 版本】

V94 integrated execution fix2, 2026-07-19.

【当时在解决什么问题】

The controller could rank block/dwell operations, but it could not express that
the current action was already preferable to every remaining operation. A
pointer reaching the end passively was not the same as a selectable identity
baseline.

【当时的 insight / hypothesis】

“Do nothing now” must be a first-class action in both policy selection and value
supervision. Without it, more execution is mandatory and capacity/dwell cannot
be interpreted as task-owned decisions. The stopping preference should be a
prior in the candidate distribution, not a surrogate execution-cost loss.

【具体做法】

- Added identity/no-op/terminal as a fixed-shape candidate from the first
  decision.
- Defined it as exact preservation of the committed action.
- Compared its physical value target with each remaining legal operation.
- Masked value supervision to remaining blocks plus terminal; already passed
  blocks no longer received target budget.
- Applied a terminal prior of `0.25` through a logit offset, leaving ordinary
  task/value gradients able to overcome it.
- Logged terminal prior, occupancy, target/predicted margin, hard fraction and
  identity error.

【观察到的结果】

- With three blocks and two dwell choices, equal candidate scores yielded an
  initial terminal probability of about
  `0.25 / (6 + 0.25) = 4%`.
- Static/dynamic tests confirmed fixed shapes, explicit hard behavior,
  action/value gradients and exact action-level identity.
- By batch 2020, operation versus accumulated terminal mass was about
  `0.584 / 0.416`, while the value reader retained nontrivial ranking quality.
- The hard route remained conservative and did not actively exit in the
  reported window; validation evidence was still missing.

【当时的解释】

The prior made early exit costly without making it impossible. More importantly,
the candidate graph could now compare continuation with the actual committed
baseline through ordinary gradients instead of using an artificial compute
penalty to force stopping.

【后来怎么看（截至冻结终点）】

Structurally retained. The batch-2020 log exposed two diagnostic refinements:
terminal occupancy was not the same as a new-exit probability, and the velocity
target's identity check was not numerically canonical even though the action
candidate itself was exact. Later history did not promote this V94 controller
into a donor architecture or establish isolated task benefit.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；committed-state candidate contract 保留，V94 controller 的任务收益 未被独立验证且实现已被替代。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 214-234 and 257-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.25 / (6 + 0.25) = 4%`、`0.584 / 0.416`。

【关联】

CI-0015, committed baseline, neutral execution and later optional/null routing.

### CI-0027 — Soft optimization and hard execution are different policies

【时间 / 版本】

V94 execution fix1/fix2, 2026-07-19.

【当时在解决什么问题】

Training mixed all legal candidate actions with soft probabilities, while
validation/deployment selected an argmax candidate. The value reader learned
useful pairwise ordering but not a reliably sharp top-one decision.

【当时的 insight / hypothesis】

A relaxation is not an implementation detail once it changes the realized
action. Until the soft distribution is sharp or explicitly distilled, hard
argmax must be treated as a separate policy and evaluated by matched-noise
ablation, not silently substituted at deployment.

【具体做法】

- Measured soft/hard dwell, route, entropy, maximum probability, pairwise
  accuracy and top-one accuracy.
- Changed the primary validation policy to soft execution.
- Kept hard and neutral policies as explicit matched-noise ablations.
- Added full-capacity and three-basis-reduction ablations to separate selection
  benefit from capacity.

【观察到的结果】

- Before fix2, soft/hard dwell was roughly `1.167 / 1.542` and route-next
  `0.257 / 0.333`; hard execution performed about 32% more dwell.
- Value pairwise accuracy reached roughly `0.82-0.87`, while top-one accuracy
  was only around `0.52-0.58` and selection max probability near `0.51`.
- In fix2 through batch 2020, soft/hard route became closer, but soft dwell still
  required conditioning on operation mass before comparison.
- No epoch validation was yet available to establish whether learned soft,
  learned hard, neutral or full-capacity execution produced the best action.

【当时的解释】

The controller could possess ranking information without supporting stable
argmax deployment. Pairwise accuracy and nonzero gradients were therefore
insufficient evidence for a hard policy.

【后来怎么看（截至冻结终点）】

Retained as a methodological rule. The implementation had aligned primary
validation with training, but the actual benefit of either execution policy
remained an experimental unknown.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；policy mismatch repaired in the default path, usefulness unverified.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 206-234 and 235-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1.167 / 1.542`、`0.257 / 0.333`。

【关联】

CI-0019, candidate-coordinate invariance, deployment parity and matched
interventions.

### CI-0028 — A continuous capacity gate is representation mass, not compute rank

【时间 / 版本】

V93-V94 capacity reinterpretation, 2026-07-19.

【当时在解决什么问题】

Logs described `effective_rank`, `selected_rank`, `removed` and
`cost_audit` as though the controller pruned real computation. In the active
native path, all 32 bases were still evaluated and continuously scaled.

【当时的 insight / hypothesis】

Continuous attenuation can be useful representation control, but it is not
discrete rank selection or a throughput claim. If task loss contains no
identifiable reason to remove harmless capacity, the natural optimum may be full
capacity even with healthy gradients.

【具体做法】

- Traced the native path's forced `binary_group_selection=False`.
- Reinterpreted capacity as `capacity_gate_mass` and
  `effective_basis_mass`.
- Kept execution cost audit-only rather than coercing compression.
- Added full-capacity and basis-reduction ablations instead of inferring benefit
  from the gate value alone.

【观察到的结果】

- The repaired controller moved transiently to roughly `31.7/32` and later
  returned near `32/32`.
- Capacity and operator-basis gradients were nonzero, unlike the detached old
  run.
- Old `depth=8` behavior was a four-group quantization boundary
  (`0/8/16/24/32`), not evidence for a healthy learned low rank.
- Fix2 at batch 2020 still had about `31.72/32` effective basis mass despite a
  live capacity gradient.

【当时的解释】

The repaired result was no longer a dead controller; it was a task objective
whose easiest solution used almost all available capacity. Calling it an
actual-rank scheduler would confuse representation attenuation with realized
compute.

【后来怎么看（截至冻结终点）】

Retained. Meaningful compute saving would require explicit eval-time pruning and
equivalence checks. The segment deliberately did not reintroduce a cost loss
merely to make the reported rank smaller.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；semantic correction completed, compression benefit unverified.

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；推断/未决
- 对话：visible messages 167-176, 195-205, 206-239 and 257-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`effective_rank`、`selected_rank`、`cost_audit`、`capacity_gate_mass`、`effective_basis_mass`。
- 直接量值/表达式（卡内原文）：`0/8/16/24/32`。

【关联】

CI-0004, CI-0005, CI-0008 and later capacity/shortcut discussions.

### CI-0029 — Aggregate RMSE and error ratios can reverse the apparent story

【时间 / 版本】

Cross-run V93/V94 validation audit, 2026-07-19.

【当时在解决什么问题】

A low full-action RMSE and a rising tail/first ratio were being read as direct
evidence that all action components improved or that tail had regressed.

【当时的 insight / hypothesis】

Aggregates and ratios must be decomposed before causal interpretation. A ratio
can worsen because its denominator improves, and a many-dimensional component
can numerically dilute a critical one-dimensional component.

【具体做法】

- Reconstructed full RMSE approximately as
  `sqrt((6 * arm_rmse^2 + gripper_rmse^2) / 7)`.
- Reported arm, gripper, first and tail absolute values beside the aggregate and
  ratio.
- Compared aligned epochs instead of only best aggregate checkpoints.

【观察到的结果】

- V94 structure repair epoch two had full RMSE `0.09654`, but gripper RMSE
  remained `0.17629`.
- Its tail/first ratio rose from about `2.63` to `3.88`, while tail itself
  improved from `0.12365` to `0.11243`; first improved much faster.
- The old V94's high event accuracy coexisted with severe decoded event
  over-triggering.

【当时的解释】

The aggregate was arm-dominated and the ratio described relative progress, not
absolute tail deterioration. Both values were valid statistics but insufficient
standalone diagnoses.

【后来怎么看（截至冻结终点）】

Retained. This reinforced the earlier finding that aggregate action metrics
cannot certify gripper or horizon behavior.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 206-219 and 235-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`sqrt((6 * arm_rmse^2 + gripper_rmse^2) / 7)`。

【关联】

CI-0010, horizon bands, rare-event evaluation and later normalized/physical
metric discussions.

### CI-0030 — A future-latent objective can be healthy in variance but weak in dynamics

【时间 / 版本】

V94 rollout objective audit, 2026-07-19.

【当时在解决什么问题】

Physical flow improved rapidly while rollout dynamics and milestone targets
changed slowly, auxiliary share grew and sampled tail error remained much larger
than the first horizon.

【当时的 insight / hypothesis】

A stop-gradient future-latent residual is not the same target as deployed
physical-action tail. Preserving overall variance does not prove that the model
has learned the magnitude or timing of state change.

【具体做法】

- Compared physical flow, rollout raw losses, exact weighted contributions,
  rollout standard-deviation ratio, step-norm ratio and sampled tail RMSE.
- Removed duplicated rollout-delta objectives in the active V94 script.
- Lowered the old rollout auxiliary budget and made temporal position weighting
  less biased.
- Deferred further reweighting until matched validation/ablation evidence
  existed.

【观察到的结果】

- Across old runs, physical flow fell roughly 92-95%, while rollout dynamics and
  milestones improved far less.
- In one exact ledger, auxiliary share grew toward half of the total as the
  action objective fell, even though the early epoch was action dominated.
- Fix2 through batch 2020 had rollout `std_ratio≈0.91` but
  `step_norm_ratio≈0.416`: distributional spread remained, while predicted
  change magnitude was conservative.
- Fix2's rollout group was about 13% of total loss at that point and still lacked
  epoch validation.

【当时的解释】

The rollout was not simply collapsed; it was smoother and lower-amplitude than
the target dynamics. More weight could force a proxy harder without proving an
improvement in sampled physical tail.

【后来怎么看（截至冻结终点）】

Partly retained as a target-alignment lesson. Duplicate charging was fixed, but
the original future-latent objective was later replaced by the Flow-DINO/JEPA
target family without a matched experiment isolating its utility.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；目标对齐原则保留，原 future-latent objective 未被单独验证。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 167-176, 206-219, 228-239 and 257-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。

【关联】

CI-0022, long-horizon supervision, later Teacher/S/W targets and sampled-tail
validation.

### CI-0031 — The auxiliary event head is not the decoded gripper policy

【时间 / 版本】

V94 gripper/event audit, 2026-07-19.

【当时在解决什么问题】

Event-head accuracy looked strong while its F1, decoded gripper events and
continuous gripper behavior disagreed.

【当时的 insight / hypothesis】

An auxiliary classifier and the final continuous action decoder are separate
consumers with separate calibration. Consistency loss and shared features do not
make one a faithful proxy for the other.

【具体做法】

- Logged decoded gripper precision/recall/F1 and event-count ratio separately
  from event-head binary/open/close metrics.
- Compared the two paths across epochs rather than using event accuracy alone.
- Reduced the event head's auxiliary weight in fix2 instead of treating its
  accuracy as a deployment objective.

【观察到的结果】

- In the repaired old run, decoded gripper F1 improved from `0.266` to
  `0.388`, and decoded event count moved from 5085 to 1451 for a target of
  1357.
- Over the same interval, event-head F1 fell from `0.151` to `0.078` and its
  predicted event count fell from 1544 to 518.
- Event-head accuracy around `0.95` was dominated by the hold class.
- The badly detached run over-triggered decoded events by roughly four to five
  times despite apparently high auxiliary accuracy.

【当时的解释】

The classifier was becoming conservative toward hold while the continuous
decoder improved for different reasons. Treating event-head accuracy as gripper
closure would have reversed the actual conclusion.

【后来怎么看（截至冻结终点）】

Retained. Whether the head supplied useful representation or merely spent loss
budget was still an experimental question, but its metrics could no longer stand
in for decoded behavior.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；semantic separation completed, auxiliary utility unresolved.

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 167-176, 206-219 and 235-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。

【关联】

CI-0010, CI-0022, rare-event class imbalance and later event-boost semantics.

### CI-0032 — Run directories and checkpoints are part of model identity

【时间 / 版本】

V94 fix2 experiment-hardening, 2026-07-19.

【当时在解决什么问题】

One default output directory had been reused across incompatible group counts,
detach settings, gradient scales and source implementations. JSONL was appended
and `latest.pt` could be overwritten without checking compatibility.

【当时的 insight / hypothesis】

Source and configuration provenance must be enforced at the write boundary, not
reconstructed only after a suspicious run. An output directory is a typed
experimental object, not a generic folder.

【具体做法】

- Added a run manifest containing source/config/data fingerprints.
- Required explicit resume before appending to an existing run.
- Rejected incompatible manifests instead of mixing epochs or checkpoints.
- Changed the default output directory to
  `runs/v94_integrated_execution_fix2`.
- Logged the active source hashes so attached logs could be checked against the
  workspace.

【观察到的结果】

- Historical V94 artifacts under one directory represented several incompatible
  execution graphs.
- The current fix2 log's six source hashes matched the then-current workspace.
- Manual use of a fresh directory had prevented one recent run from being
  contaminated, but correctness had previously depended on remembering that
  convention.

【当时的解释】

Without a manifest, an apparently smooth multi-epoch curve could cross
implementation boundaries. Post-hoc log analysis could not repair an overwritten
checkpoint or reliably separate appended epoch records.

【后来怎么看（截至冻结终点）】

Retained. This extended experiment identity from “read the header” to “refuse an
invalid write/resume.”

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；fix2 boundary implemented and tested.

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 214-234 and 235-260。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`latest.pt`、`runs/v94_integrated_execution_fix2`。

【关联】

CI-0001, checkpoint ABI, schema resume rejection and source fingerprints.

### CI-0033 — Numerical identity diagnostics must reuse the canonical computation

【时间 / 版本】

V94 fix2 terminal audit through batch 2020, 2026-07-19.

【当时在解决什么问题】

`terminal_identity_error` grew from roughly `3.5e-4` to `3.0e-3`, which
looked like evidence that the new no-op candidate was modifying the action.

【当时的 insight / hypothesis】

An identity probe is only meaningful if both sides share the exact numerical
path. Recomputing an equivalent tensor with a different BF16 GEMM batch layout
can measure kernel/layout differences instead of a contract violation.

【具体做法】

- Traced terminal construction to an exact `action.clone()`.
- Traced baseline velocity decoding at shape `[B,...]` and candidate velocity
  decoding after flattening to `[B*C,...]`.
- Compared the error with the candidate target spread.
- Proposed reusing the already computed baseline velocity for the terminal slot
  rather than decoding it again.
- Reinterpreted `terminal_probability` as accumulated occupancy and requested
  a separate active-exit statistic.

【观察到的结果】

- Action-level terminal identity was exact.
- Velocity-level discrepancy reached about `0.003`, roughly 3% of target
  spread, while the two decodes used different BF16 matrix layouts.
- The discrepancy was non-negligible as a target impurity but did not explain a
  controller collapse or justify stopping the run.
- Reported terminal mass mixed new exits with pointer mass that had entered
  terminal on previous decisions.

【当时的解释】

The no-op contract itself was not disproven; the diagnostic created a
non-canonical second decode. The correct repair was to share the baseline tensor
and separate occupancy from transition probability.

【后来怎么看（截至冻结终点）】

Retained as a precise implementation debt and a correction to the audit tool's
initially severe identity warning. The frozen archive does not contain a later
independent closure record for this exact diagnostic, so its historical repair
state must remain unknown rather than “pending.”

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；根因已定位，冻结档案未记录该历史 diagnostic 的独立修复闭环。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 257-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`terminal_identity_error`、`action.clone()`、`terminal_probability`。

【关联】

BF16 equivalence, neutral zero semantics, diagnostic false positives and
terminal ownership.

### CI-0034 — Throughput numbers need explicit timing boundaries

【时间 / 版本】

V94 fix2 runtime audit, 2026-07-19.

【当时在解决什么问题】

The longer compact log made the new run feel slower, and a reported
`2.679 s/batch` was being treated as a complete epoch or pure-GPU performance
measurement.

【当时的 insight / hypothesis】

Training interval time, validation sampling, diagnostic work and epoch overhead
are different clocks. A correct aggregate timer can still be insufficient for
locating a bottleneck.

【具体做法】

- Reconstructed `2.679` as the median of ten 20-batch interval averages from
  batches 20-200.
- Enumerated what the timer included: data wait, transfer, forward/loss,
  backward, gradient diagnostics, clipping, optimizer, metric accumulation and
  amortized log synchronization.
- Separated validation, execution ablations, checkpoint/JSON writes and full
  epoch wall time.
- Inspected the logging and execution implementation for hidden work.

【观察到的结果】

- Warmup, transition and full-progress medians were approximately
  `2.679`, `2.725` and `2.718 s/batch`; the observed training path had not
  materially slowed.
- Roughly 2972 batches at `2.7 s` already implied about 134 minutes of training
  before validation.
- Full gradient diagnostics traversed overlapping large modules every batch even
  though they were printed every 20 batches.
- Roughly 700 GPU scalars were accumulated every batch, including many inactive
  values filtered only at epoch end.
- Soft execution serially evaluated candidate operations; the neutral ablation
  still paid for a complete soft candidate chart.
- Thirty-two additional validation samples were not timed separately and were
  mislabeled as non-sampling time.

【当时的解释】

The longer text output was not the observed slowdown. The principal epoch cost
was the number of training batches, while diagnostics and serial candidate
evaluation were real but distinct efficiency debts.

【后来怎么看（截至冻结终点）】

Retained. The safe first optimization was to reduce observation overhead without
changing model math; neutral fast paths and candidate vectorization required
stronger equivalence tests because they touched execution semantics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；diagnostic and timing debts identified, no performance modification in this segment.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预
- 对话：visible messages 240-260。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`2.679 s/batch`、`2.718 s/batch`。

【关联】

Lifecycle accounting, validation ablations, logging overhead and future
throughput probes.

### CI-0035 — Structural tests establish legality, not task usefulness

【时间 / 版本】

V94 integrated execution fix2, 2026-07-19.

【当时在解决什么问题】

The no-op/terminal graph, soft primary policy, manifest and diagnostics passed
targeted tests, creating a temptation to declare the execution architecture
closed before the first validation.

【当时的 insight / hypothesis】

Tests can prove shape, mask, gradient, identity, compatibility and ledger
invariants. Only matched intervention on real checkpoints can show that learned
execution improves the task relative to neutral or full-capacity alternatives.

【具体做法】

- Ran 24 execution/decoder checks, seven loss/logger/manifest checks, five log
  parser checks, compilation, diff inspection and wrapper preflight.
- Started a fresh fix2 run with terminal, soft primary evaluation and explicit
  execution ablations.
- Interpreted batch-2020 metrics separately from unavailable epoch validation.

【观察到的结果】

- Training remained finite; the loss ledger closed; all important gradient paths
  were nonzero.
- Value correlation was about `0.64`, pairwise accuracy `0.83`, top-one
  accuracy `0.62`, and predicted spread about two thirds of target spread.
- Main action loss remained dominant and no controller collapse appeared.
- Capacity stayed near full, rollout change magnitude was conservative, hard
  exit was very cautious and no soft/hard/neutral/full-capacity validation result
  yet existed.

【当时的解释】

The experiment supported “the repaired graph is trainable and internally
nontrivial,” not “the controller improves action quality.” This was a narrower
and more defensible claim than the earlier static completion language.

【后来怎么看（截至冻结终点）】

Retained as another explicit correction to a premature closure claim. Later V94
behavior did not convert the structural test suite into proof that learned
execution itself was useful, and the controller was not retained as a donor.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；测试只建立 legality，未建立版本级 usefulness。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录
- 对话：visible messages 220-234 and 235-264。
- 段级工件：见 Segment 003 boundary 的 `Primary historical artifacts`。

【关联】

CI-0012, CI-0019, matched ablation and the later project-wide functional-closure
standard.

## Segment 003 synthesis

This segment moved the V94 investigation from “is the controller connected?” to
“does the controller own a distinguishable, deployable decision?” It also
separated several notions that had repeatedly been collapsed:

```text
falling training loss        != ownership learning
nonzero gradient             != beneficial control
pairwise ranking             != stable top-one policy
continuous basis mass        != realized compute rank
soft training policy         != hard deployment policy
terminal occupancy           != active exit probability
auxiliary event accuracy     != decoded gripper behavior
full-action RMSE             != every component is healthy
passing structural tests     != matched task benefit
```

The integrated fix2 repaired real graph defects: it introduced a committed-state
candidate, aligned the primary policy, hardened run provenance and made capacity
semantics more honest. Its early log then supplied an equally important
restraint: through batch 2020 the graph was trainable and non-collapsed, but
there was still no validation evidence that learned execution beat neutral or
full-capacity behavior. That distinction is the durable outcome of the segment.

## Segment 004 — Artifact checkpointing and mixed-precision lifecycle closure

### Segment boundary

```text
Date:                 2026-07-20 to 2026-07-21
Visible messages:     265-318
Starts with:          remote-monitoring and overdue Git checkpoint discussion
Ends with:            fix3 epoch-one comparison after the dtype repair
Primary historical artifacts:
  .gitignore
  branch codex/v94-latent-ownership-execution
  commit b2c1fa8
  scripts/current_v94_latent_ownership_execution.sh
  clearvla/policy/system.py
  clearvla/policy/time_domain_mmdit.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  tests/test_time_domain_mmdit.py
  attached fix2/fix3 V94 logs
```

### CI-0036 — A research checkpoint must exclude runtime state and disclose verification gaps

【时间 / 版本】

V94 source checkpoint, 2026-07-20 to 2026-07-21.

【当时在解决什么问题】

The active branch had not been uploaded for several days and contained hundreds
of modified and untracked files, including a local source backup, research
artifacts and run-adjacent data.

【当时的 insight / hypothesis】

A useful source checkpoint is a curated recovery boundary, not a snapshot of
everything visible in the working tree. It must separately establish inclusion
scope, secret/binary exclusion, source status and what was or was not actually
tested.

【具体做法】

- Audited the branch, upstream relation, tracked changes, deletions, untracked
  files and large paths before staging.
- Found that `clearvla_backup/` was ignored but the real local copy was named
  `clearvla_before/`.
- Extended ignore rules for the actual backup, environment/secret patterns,
  checkpoints, outputs, experiment trackers and common model binaries.
- Kept the project log-audit skill versioned while excluding logs, weights,
  caches, data and raw backup material.
- Used an explicit staging scope rather than `git add -A`.
- Created and pushed branch `codex/v94-latent-ownership-execution` at commit
  `b2c1fa8`.

【观察到的结果】

- The initial worktree contained 190 modified tracked files, five tracked
  deletions and 235 untracked files; the backup alone contained about 406 files.
- The staged checkpoint contained 231 reviewed files and no detected
  log/data/weight/backup/environment/secret patterns.
- Python compilation passed, but the local Windows environment lacked
  PyTorch/pytest and therefore did not execute model tests.
- A later real validation run still exposed a mixed-precision graph failure,
  demonstrating that a clean commit and compilation were not runtime closure.

【当时的解释】

The checkpoint succeeded as source preservation, but its verification boundary
had to remain explicit. Version control made the state recoverable; it did not
certify the unexecuted CUDA/deployment path.

【后来怎么看（截至冻结终点）】

Retained. The immediately following dtype failure is the strongest evidence in
this segment for never collapsing “committed,” “compiled” and “validated” into
one status.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；checkpoint completed, CUDA verification explicitly absent at commit time.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 266-287 and 288-303。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`clearvla_backup/`、`clearvla_before/`、`codex/v94-latent-ownership-execution`、`b2c1fa8`。

【关联】

CI-0001, CI-0032, experiment recovery and later checkpoint ABI discipline.

### CI-0037 — Mixed-precision state needs semantic dtype ownership

【时间 / 版本】

V94 fix2 validation crash and fix3 dtype contract, 2026-07-21.

【当时在解决什么问题】

Fix2 completed roughly 2972 training batches, then failed on the first
deploy-style validation sample:

```text
index_add_(): self (BFloat16) and source (Float)
must have the same scalar type
```

Earlier BF16 failures had been repaired at individual operations, but the
execution state still inherited dtype from whichever neighboring tensor created
it.

【当时的 insight / hypothesis】

Dtype is part of tensor ownership. Action/features may use autocast BF16, while
scores, probabilities, entropy, occupancy and pointer mass are numerical-policy
state and should remain FP32. A local cast at the failing `index_add` would
hide the boundary defect rather than define it.

【具体做法】

- Traced model parameters as FP32, visual inputs as BF16, policy actions in
  training as FP32 and deployment sampling actions as visual-dtype BF16.
- Located pointer creation through `action.new_zeros` and probability creation
  through FP32 score tensors.
- Traced silent promotion in ordinary addition followed by strict dtype failure
  at `index_add`.
- Fixed execution scores, softmax outputs, entropy, probabilities, occupancy and
  pointer recurrence to FP32.
- Kept action/features in BF16/autocast and cast probabilities only at the
  unique action-mixing boundary.
- Covered soft, hard and neutral policies and ensured subsequent pointer updates
  could not inherit BF16 again.

【观察到的结果】

- Training did not reveal the defect because its policy action/pointer was
  usually FP32; deploy sampling inherited BF16 from visual input.
- Existing tests separately covered BF16, autocast and dynamic soft execution,
  but not their real composition:
  `FP32 parameters + CUDA BF16 autocast + dynamic soft policy + deploy sample`.
- Thirty-three related tests passed after the contract repair.
- A real server run printed
  `deploy_sampling=pass dtype=bfloat16 steps=5`, completed validation and
  continued into epoch two without dtype or non-finite failure.

【当时的解释】

The failing line was an alarm, not the origin. The origin was that action-domain
and probability-domain tensors had no explicit dtype owners, so train and
deployment happened to construct the same logical state in different numerical
domains.

【后来怎么看（截至冻结终点）】

Retained. This repair closed the execution dtype boundary rather than the
individual error site, and the real CUDA path confirmed it.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；确定性 dtype 故障已修复并经真实 CUDA 验证。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 288-306 and 310-318。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`index_add`、`action.new_zeros`。

【关联】

BF16/FP32 ownership, deployment parity, strict scatter/index operations and
later numerical-boundary audits.

### CI-0038 — Deployment sampling must be a pre-training lifecycle gate

【时间 / 版本】

V94 fix3 fail-fast repair, 2026-07-21.

【当时在解决什么问题】

The train graph ran a complete epoch before the first validation/deployment path
was exercised. Because validation preceded epoch JSON/checkpoint writing, the
failure discarded both validation evidence and the normal epoch-one recovery
point.

【当时的 insight / hypothesis】

A path that is required for experiment completion must be exercised before the
expensive phase begins. Unit coverage is not enough when the deployment graph
has a different autocast/input construction. The preflight must preserve model
mode and RNG so that the check itself does not change the experiment.

【具体做法】

- Added a one-batch, five-step, deploy-style BF16 `system.sample` preflight
  before epoch one.
- Kept the real sampling path, rather than a simplified shape-only proxy.
- Restored RNG and model train/eval state after the check.
- Required finite outputs and the same soft/hard/neutral dtype contracts.
- Defined a short smoke intended to include 20 train batches, two validation
  batches, one diagnostic batch, checkpoint creation and epoch JSON output.

【观察到的结果】

- The old fix2 failure occurred only at validation and therefore produced no
  epoch-one validation record or normal `latest.pt`.
- The repaired server run passed the preflight and then the full validation path.
- The run described as a smoke actually serialized
  `epochs=8, max_train_batches=0, max_val_batches=0` and was a formal run,
  reinforcing that the run context—not the operator's label—defines the
  experiment.

【当时的解释】

The preflight moved a deterministic lifecycle error from after hours of training
to before training. Preserving RNG and model state kept it an assertion about
the deployment graph rather than a hidden extra training intervention.

【后来怎么看（截至冻结终点）】

Retained. It solved engineering waste but did not certify model quality; the
first complete validation immediately exposed separate tail, gripper and
ablation questions.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；fail-fast lifecycle gate implemented and validated.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预
- 对话：visible messages 288-314。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`system.sample`、`latest.pt`。

【关联】

CI-0011, CI-0015, checkpoint timing and all later smoke/release gates.

### CI-0039 — Validation and checkpoint ordering can erase an otherwise complete epoch

【时间 / 版本】

V94 fix2 failure analysis, 2026-07-21.

【当时在解决什么问题】

Almost all epoch-one training completed successfully, but validation crashed
before the training loop wrote its epoch record or latest checkpoint.

【当时的 insight / hypothesis】

Lifecycle ordering is part of failure recovery. “Validate before checkpoint” can
be attractive for keeping only complete epochs, but it also turns a validation
bug into loss of a long, finite training state unless a recoverable
pre-validation checkpoint exists.

【具体做法】

- Traced the runtime order from training completion into validation, JSONL write
  and checkpoint save.
- Distinguished the absent normal epoch checkpoint from the completed in-memory
  training work.
- Prioritized deploy-sampling preflight before rerunning.

【观察到的结果】

- Fix2 reached the end of epoch-one training with improving flow and controller
  metrics.
- The first validation sample failed, so there was no validation row, epoch JSON
  or normal epoch-one `latest.pt`.
- The expensive training state could not answer any full/first/tail, decoded
  event or execution-ablation question.

【当时的解释】

The run failed as an experiment even though its train loop was healthy. A later
preflight prevented recurrence, but the checkpoint lifecycle itself remained an
important recovery consideration.

【后来怎么看（截至冻结终点）】

Retained as a design warning. This segment fixed early detection; it did not
record a separate pre-validation recovery checkpoint.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；failure prevented by preflight, recovery-order tradeoff remained.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 288-303。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`latest.pt`。

【关联】

Checkpoint atomicity, exact resume, validation gates and long-run recovery.

### CI-0040 — Runtime closure only restores the right to interpret model quality

【时间 / 版本】

V94 fix3 epoch-one validation, 2026-07-21.

【当时在解决什么问题】

After the dtype repair, the first complete run could finally produce validation.
The question was whether the fix had materially worsened performance or solved
the outstanding model problems.

【当时的 insight / hypothesis】

An engineering repair should first be checked for behavioral regression, but
successful execution does not imply that pre-existing representation or target
problems have improved.

【具体做法】

- Verified source hashes, full epoch/validation completion, finite behavior and
  exact loss-ledger closure.
- Compared the result with the latest compatible V94 validation rather than with
  a crashed fix2 run.
- Decomposed full, first, first-eight, tail, arm, gripper and decoded-event
  metrics.
- Kept execution ablations provisional because only eight batches (about 4%
  coverage) were sampled and compact output omitted the same-batch primary
  baseline.

【观察到的结果】

- Full RMSE was `0.09829` versus `0.09654` previously, a 1.8% difference;
  tail RMSE was `0.11461` versus `0.11243`, and gripper RMSE slightly
  improved from `0.17629` to `0.17416`.
- Tail/first improved slightly from `3.876` to `3.766` but remained far
  above the deployment gate.
- Decoded gripper event ratio rose from `1.069` to `1.632`; recall and F1
  rose slightly while precision fell, indicating a more aggressive policy with
  many more false positives.
- Rollout dynamics and milestones still improved far more slowly than the action
  objective.
- Capacity remained near `31.81/32`, while controller ranking metrics stayed
  healthy.

【当时的解释】

The dtype repair did not cause a broad accuracy collapse. It made the run
measurable again, after which the same long-horizon imbalance and event-policy
misalignment remained visible. The largest behavioral shift was gripper
aggressiveness, not aggregate RMSE.

【后来怎么看（截至冻结终点）】

Retained as the proper separation between engineering and modeling outcomes.
The conversation next chose to revisit top representation rather than continue
weight-level patches.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；runtime fault closed, model-quality issues carried forward.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录
- 对话：visible messages 310-318。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。

【关联】

CI-0029, CI-0030, CI-0031 and the following top-representation redesign.

### CI-0041 — Sparse ablations need their own matched primary baseline

【时间 / 版本】

V94 fix3 execution ablation, 2026-07-21.

【当时在解决什么问题】

Compact validation output printed hard, neutral, full-capacity and basis-reduced
RMSEs that appeared substantially lower than the full primary RMSE.

【当时的 insight / hypothesis】

A sparse ablation subset cannot be compared directly with the full validation
set. Every intervention needs the unchanged primary policy evaluated on the same
examples, noise and sampling budget.

【具体做法】

- Traced execution ablations to eight validation batches, about 4% coverage.
- Located a same-batch `execution_ablation_primary_full_rmse` in the runtime.
- Noted that compact nohup output omitted this matched baseline even though JSONL
  was expected to retain it.
- Restricted interpretation to within-subset differences that did not require
  the missing primary.

【观察到的结果】

- Compact values were approximately hard `0.09285`, neutral `0.08311`,
  full-capacity `0.07996` and three-basis `0.08046`.
- Full-capacity versus three-basis differed by only `0.00050` on the sampled
  subset.
- Without the matched primary value, the log could not establish that learned
  soft execution was worse than neutral or full capacity.

【当时的解释】

The apparent improvement could be mostly subset difficulty. Coverage and a
paired primary were part of the causal statistic, not optional metadata.

【后来怎么看（截至冻结终点）】

Retained. This explicitly prevented a sparse diagnostic from becoming an
architecture verdict.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；interpretation corrected, compact logger still lacked the matched value.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预
- 对话：visible messages 310-318。
- 段级工件：见 Segment 004 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`execution_ablation_primary_full_rmse`。

【关联】

CI-0018, CI-0019, CI-0024 and later matched zero/shuffle interventions.

## Segment 004 synthesis

This segment supplied a compact example of why ClearVLA eventually required
producer/consumer and lifecycle contracts rather than local bug fixes:

```text
source checkpoint complete
    does not imply CUDA/deploy path tested

train epoch complete
    does not imply validation or recoverable epoch artifact complete

BF16 unit path + autocast unit path + dynamic-policy unit path
    does not imply their deployed composition was tested

deploy graph fixed
    does not imply long-horizon or gripper behavior fixed
```

The durable dtype repair assigned FP32 ownership to policy-distribution state and
BF16/autocast ownership to action/features, then exercised their real composition
before training. Once runtime evidence became available, the result was modest
and specific: no broad regression, persistent tail imbalance, more aggressive
gripper events and still-unproven execution benefit. Engineering closure restored
the ability to run the experiment; it did not answer the representation question
that begins the next segment.

## Segment 005 — From the pseudo-top bottleneck to Flow-DINO/JEPA and stateless stage

### Segment boundary

```text
Date:                 2026-07-21
Visible messages:     319-375
Starts with:          request to reconsider the top representation
Ends with:            V95 hierarchical Flow-DINO/JEPA implementation handoff
Primary historical artifacts:
  clearvla/policy/trunk.py
  clearvla/policy/flow_dino_evidence.py
  clearvla/policy/trunk_primitives.py
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/controller.py
  scripts/current_v95_flow_dino_jepa.sh
  tests for Flow-DINO, directed attention, loss, logging and resume
  I-JEPA, V-JEPA, MC-JEPA, RAFT and SEA-RAFT design references
```

### CI-0042 — The V94 “top representation” was an early pooled bottleneck

【时间 / 版本】

V94 top-representation audit, 2026-07-21.

【当时在解决什么问题】

Short-horizon action, motion and candidate ranking were learnable, while tail,
rollout and gripper semantics remained weak. The user questioned whether the
supposed top representation was structurally appropriate.

【当时的 insight / hypothesis】

The active `z` was not a deep top-level semantic state. It pooled
pre-attention seed memory into one 64-dimensional vector, expanded it to 512 and
broadcast it across 24 horizon tokens. This protected against noisy-action
feedback but also excluded later reasoning and forced task, phase, event and
future structure through one common carrier.

【具体做法】

- Traced `owned_intent_memory` to a point before trunk canvas
  self-attention.
- Traced the layer organizer's values back to repeated reads of the same clean
  intent memory rather than the actual per-layer representations.
- Found two global intent aggregators feeding one condition without a clear
  semantic division.
- Traced diffusion time both into `global_condition` and separately into the
  controller.
- Compared source flow with `z_zero` and `z_shuffle` interventions.

【观察到的结果】

- The effective path was:
  pre-attention task/state/history/executed/proposal/visual
  `→` pooled condition `→ 64D z → 512D` broadcast.
- Nominal layer-depth scan values were learned queries rereading one early
  memory, while the layer axis mostly acted as selector information.
- `z_zero_cond_delta≈0.584` but batchwise
  `z_shuffle_cond_delta≈0.015`.
- Local motion F1 and controller ranking were relatively healthy while event
  F1, tail/first and rollout learning lagged.

【当时的解释】

The representation carried a strong useful common bias but weak sample- and
horizon-specific structure. Avoiding one shortcut by taking the representation
too early had created an information bottleneck and a misleading “top” name.

【后来怎么看（截至冻结终点）】

This diagnosis motivated V95's structured evidence and directed information
flow. Later V95-V98 evidence repeatedly supported the bandwidth concern, but
the bundled redesigns never isolated the pooled bottleneck as the dominant
task-performance cause.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；source diagnosis 保留，独立 behavioral causality 未被隔离。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 319-327。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`owned_intent_memory`、`global_condition`、`z_zero`、`z_shuffle`。

【关联】

Single-vector bottlenecks, public common mode, layer ownership and later
structured G/S/W/P representations.

### CI-0043 — Discussion permission and implementation permission are different states

【时间 / 版本】

V95 design initiation, 2026-07-21.

【当时在解决什么问题】

After the user said the top representation also needed to change, the assistant
began describing implementation sequencing before the user had authorized code
changes.

【当时的 insight / hypothesis】

Agreement that a subsystem is wrong does not authorize immediate mutation.
Architecture exploration, design adoption and implementation start are separate
conversation states.

【具体做法】

- The user explicitly stopped the premature implementation move.
- The assistant confirmed that only read-only source review and planning had
  occurred.
- Discussion continued through representation goals, mask location, unified
  routing and evidence-reader contracts.
- Code work began only after the later explicit `来开始吧`.

【观察到的结果】

The correction did not require reverting source because no edit had yet been
made, but it changed the workflow from implied implementation to explicit design
alignment.

【当时的解释】

“This must eventually change” was a statement about desired architecture, not a
command to edit immediately.

【后来怎么看（截至冻结终点）】

Retained as a collaboration and research-governance rule, especially for
high-risk top-level rewrites.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录
- 对话：visible messages 319-342。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。

【关联】

Mandatory subsystem familiarity, plan approval and later controlled schema
implementation.

### CI-0044 — One trainable path can still require typed roles and readouts

【时间 / 版本】

V95 top design, 2026-07-21.

【当时在解决什么问题】

An initial diagram appeared to split world prediction and action generation into
two heads. The user wanted one path so that learned world structure could not
become another optional side module.

【当时的 insight / hypothesis】

One semantic path does not mean one undifferentiated numerical head. Content,
motion, noisy action and masked future can share a directed DiT backbone while
retaining typed token roles and small role-specific readouts.

【具体做法】

- Defined token families for visible context, observed motion, noisy action and
  masked future queries.
- Set the conceptual direction
  `context → motion → action → future`.
- Kept the future teacher outside the trainable path.
- Assigned action velocity, future latent and flow/uncertainty to lightweight
  typed readouts over one shared backbone.
- Defined `Z_top` as structured context plus motion evidence rather than a
  separate pooled output.

【观察到的结果】

The proposed single path could let:

- action tokens read observed content/motion;
- future tokens read the action and predict consequences;
- future prediction gradients improve evidence used by the action;
- teacher content remain unavailable to the action path.

【当时的解释】

The previous failure was fragmented ownership, not the mere existence of
different tensor types. Removing type boundaries would have created leakage and
semantic ambiguity rather than unity.

【后来怎么看（截至冻结终点）】

Implemented in V95 as a directed shared path with typed evidence/readouts. The
typed-role principle survived later architectures, while the exact V95
realization was superseded and never isolated as a performance cause.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；typed-role 原则保留，V95 realization 未获独立收益归因。

【证据与锚点】

- 类型：日志观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 328-340 and 341-353。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`Z_top`。

【关联】

Directed attention, one-path ownership, later evidence-to-action closure and
typed P2/P3 lanes.

### CI-0045 — The JEPA mask is an early information boundary, not a late loss mask

【时间 / 版本】

V95 Flow-conditioned masked JEPA design, 2026-07-21.

【当时在解决什么问题】

The user expected the mask to act near the input, while an easy implementation
could allow a trainable encoder or flow module to see full future content and
only mask positions at the final comparison.

【当时的 insight / hypothesis】

If any online trainable module has already consumed the target content, masking
the loss cannot create a prediction problem. Masking must precede online
adaptation/evidence compilation, and the complete future target must exist only
in a frozen/no-grad teacher coordinate system.

【具体做法】

- Placed masking after cached DINO extraction but before any trainable online
  adapter, flow evidence or DiT.
- Represented hidden positions with position/type mask queries, not their real
  DINO values.
- Kept complete future DINO in a frozen target path and compared only the chosen
  coordinates.
- Allowed observed-history flow but prohibited future flow computed from real
  future frames from entering the policy.
- Fixed the mask quantity and let flow statistics only influence location.
- Used detached flow statistics for discrete mask sampling so the flow model
  could not reduce loss by hiding difficult regions.

【观察到的结果】

The resulting contract distinguished:

```text
same spatial/camera/time coordinates and latent space
different visibility:
  online sees visible context + mask queries
  teacher sees complete future and supplies stop-gradient targets
```

Flow participated through warp, correlation and prediction losses rather than
only controlling which examples were scored.

【当时的解释】

The mask defined what the model was allowed to know. Fixing target count also
prevented a learned flow/mask policy from declaring hard regions uninteresting
and shrinking the task.

【后来怎么看（截至冻结终点）】

Implemented in the V95 static contract. The frozen archive contains no matched
mask ablation establishing the utility of that exact realization; later target
families retained the information-boundary principle rather than the operator.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；mask 边界原则保留，V95 realization 未获独立验证。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 328-340 and 341-353。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。

【关联】

Teacher isolation, information leakage, optional routing shortcuts and later
future-target audits.

### CI-0046 — DINO and flow should remain typed evidence until query-time fusion

【时间 / 版本】

V95 evidence-bank design, 2026-07-21.

【当时在解决什么问题】

The project needed DINO semantics and learnable motion to cooperate without
pooling both into one vector or letting one untyped addition drown out the
other.

【当时的 insight / hypothesis】

DINO content, motion/correlation, warped content and reliability are
complementary evidence. They should share aligned patch coordinates but retain
typed projections. A DiT can then reread one compiled bank at every layer using
layer-specific queries and shared selector/value semantics.

【具体做法】

- Defined per-patch content, flow, occlusion, confidence, correlation, warp and
  camera/time/space position.
- Made selector/key answer “where should this state read?” and value answer
  “what content/motion should be returned?”
- Combined the per-block cross-reader pattern with the controller's genuine
  selector/value lane separation.
- Shared evidence projectors and K/V semantics while keeping per-layer query
  projection and residual gate.
- Corrected an implementation that computed the same `2H` projection twice
  and discarded half each time by directly slicing the compatible weights.

【观察到的结果】

- The old TimeDomain reader exposed separate token arguments but derived both K
  and V from the value stream.
- V95 compiled content, motion and warp into approximately 896 evidence tokens
  and allowed each block to reread them from its current action/future state.
- Low-confidence warp content was attenuated without deleting the underlying
  semantic content lane.

【当时的解释】

Fusion at query time preserved both evidence identity and the action-dependent
question. A one-time global sum would have lost the ability to tell whether a
read came from appearance, geometry or correspondence.

【后来怎么看（截至冻结终点）】

Implemented in V95 and repeatedly re-expressed in later typed G/S/W/P paths.
The principle survived, while the exact V95 fusion path was superseded and its
isolated action utility was never established.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；typed-evidence 原则保留，V95 路径未获独立 action 归因。

【证据与锚点】

- 类型：方案/设计记录；推断/未决
- 对话：visible messages 337-353。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。

【关联】

Selector/value ownership, repeated evidence reading, typed future dynamics and
later public/typed information conservation.

### CI-0047 — Learnable flow required a real correspondence model, not an MLP readout

【时间 / 版本】

V95 Flow-DINO implementation, 2026-07-21.

【当时在解决什么问题】

The user explicitly rejected a toy optical-flow attachment and supplied
NeuFlow, SEA-RAFT and MemFlow as possible implementation references.

【当时的 insight / hypothesis】

For the available three-frame DINO history, a SEA-RAFT-style patch flow offered
the needed combination of direct initialization, all-pairs correlation,
iterative refinement and uncertainty without importing an RGB backbone or a
long-lived memory protocol.

【具体做法】

- Adapted the SEA-RAFT design to DINO patch-grid features.
- Implemented bidirectional all-pairs correlation, correlation pyramid,
  soft-match initialization, shared recurrent refinement, uncertainty,
  occlusion, warp and forward/backward consistency.
- Kept online gradients through the refinement coordinate path instead of
  copying an implementation detach whose purpose belonged to another training
  setup.
- Derived temporal interval from dataset `history_offsets` rather than a
  hard-coded `dt=4`.
- Made flow, cycle, uncertainty and smoothness objectives bidirectionally
  symmetric.
- Removed duplicate supervision between final warp and the last refinement
  sequence element.

【观察到的结果】

- The module added about 4.36 million parameters.
- Default geometry produced a finite BF16 evidence bank of shape
  `[1,896,512]` and future queries of shape `[1,512,512]`.
- Forty-three targeted causal, gradient, teacher-isolation, BF16 and ledger tests
  passed before the later stage extension.
- V94 remained unchanged when the Flow-JEPA feature was disabled.

【当时的解释】

The design reused a mature correspondence/update pattern while changing the
input representation and lifecycle to match ClearVLA. NeuFlow's edge-oriented
RGB throughput and MemFlow's long memory would have introduced less relevant
contracts for the current history.

【后来怎么看（截至冻结终点）】

Partly retained. Later logs confirmed that architectural ancestry and a
trainable correspondence path did not prove action-relevant motion; V95/V98
realizations remained weak or action-irrelevant even though the requirement for
a real correspondence model survived.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；correspondence 要求保留，早期 realization 未证明 action utility。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 339-353。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`history_offsets`。

【关联】

Optical-flow confidence, warp targets, later Flow-DINO logs and source provenance.

### CI-0048 — Replacing a target family requires removing the old consumer, not adding a new loss beside it

【时间 / 版本】

V95 objective migration, 2026-07-21.

【当时在解决什么问题】

The new masked future-DINO target could have been layered on top of V94's weak
rollout/future-latent objective, duplicating future supervision and leaving two
incompatible semantics active.

【当时的 insight / hypothesis】

An objective replacement is complete only when the old target producer,
consumer, loss contribution and logger leave the active experiment together.

【具体做法】

- Stopped generating the old `rollout_effect_target` in the V95 path.
- Disabled the old rollout/future-latent objective family in the V95 script.
- Added masked window/future prediction, flow and later stage targets to the
  exact ledger and diagnostics.
- Ensured V94 did not silently enter directed-canvas or Flow-JEPA behavior when
  the feature was off.

【观察到的结果】

Static review reported the old rollout family absent from both V95 computation
and logging rather than merely assigned a low weight.

【当时的解释】

Keeping both families would have obscured which future semantics supplied the
gradient and recreated the duplicate-objective problem previously seen in V94.

【后来怎么看（截至冻结终点）】

Retained as a clean migration decision; the new objective's actual value was
still unknown.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；old V95-incompatible target removed from the active graph.

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；推断/未决
- 对话：visible messages 341-353 and 366-375。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`rollout_effect_target`。

【关联】

CI-0020, CI-0030, objective ownership and later Teacher target algebra.

### CI-0049 — Stateless stage inference can use history without owning hidden state

【时间 / 版本】

V95 stage design, 2026-07-21.

【当时在解决什么问题】

The assistant initially proposed persistent cross-window task memory for long
tasks. The user preferred a stateless “attachment” that remained a serious
network and learned through the main gradient.

【当时的 insight / hypothesis】

Statelessness concerns lifecycle ownership, not input impoverishment. A stage
observer can recompute stage from task, Flow-DINO evidence, robot-state history
and executed-action history without maintaining a hidden state across calls.

【具体做法】

- Rejected the initial persistent `TaskStageMemory` proposal for the active
  version.
- Considered learned stage queries with cross-attention over evidence and
  finite-history inputs.
- Kept the observer inside the main action/future gradient path, with no detach,
  artificial gradient scale, stage labels or independent pretraining.
- Used audit-only stage gradient, ablation, variance, entropy and task-shuffle
  diagnostics rather than auxiliary stage-classification loss.
- Deferred GRU/LSTM state until evidence showed that identical observable
  windows required different decisions.

【观察到的结果】

The conversation explicitly recognized the information limit: two truly
identical observations with identical task and finite history cannot be assigned
different stages by any stateless model.

【当时的解释】

The current dataset could provide three-frame visual/flow history, robot-state
history and executed actions without introducing reset, sequence sampling,
truncated-BPTT or deployment-state synchronization.

【后来怎么看（截至冻结终点）】

The stateless-with-history lifecycle survived, but the one-token stage
realization was later replaced by multi-horizon spatial evidence and then by
explicit interval organization. The exact V95 carrier is not a recovery unit.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；stateless lifecycle 保留，单 stage-token carrier 已被替代。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 354-375。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。

【关联】

Action history, Goal Tokens, phase identifiability and later stateless S
organizer work.

### CI-0050 — Local transition and long-range stage require different target granularity

【时间 / 版本】

V95 hierarchical future design, 2026-07-21.

【当时在解决什么问题】

Four dense anchors at `4/8/12/16` were too locally uniform for stage
awareness, but predicting an exact patch state at `t+48` from only a 24-step
action chunk would be underconditioned.

【当时的 insight / hypothesis】

Multiple time scales do not require a second training stage or a recurrent
model. One forward/backward can use fine action-conditioned window targets
inside the known horizon and a coarser, explicitly non-transition stage target
beyond it.

【具体做法】

- Considered a single four-anchor set `4/8/16/24` with real horizon
  embeddings.
- Refined it into three window targets at `4/12/24` and one coarse stage
  target at `48`.
- Made the stage representation/prediction precede and condition the window
  tokens without detach.
- Prohibited ground-truth future DINO from conditioning the window path.
- Used a camera-by-`2x2` spatial summary for the stage target rather than a
  global mean or exact full patch reconstruction.
- Kept one optimizer and one joint forward/backward; did not add LSTM/GRU or
  cross-batch state.

【观察到的结果】

- Window and stage used cached target grid indices `[0,2,5,11]` for
  `[4,12,24,48]`.
- The stage loss weight was `0.02` and window loss weight `0.10`.
- Production-shape forward/backward gave finite nonzero gradients to both stage
  query and stage-to-window bridge.
- The design explicitly labelled `t+48` as a coarse stage/result prior, not a
  fully action-determined physical transition.

【当时的解释】

Within 24 steps, spatial future prediction could be conditioned on the supplied
action chunk. Beyond 24, uncertainty about unobserved future actions required a
coarser semantic target rather than pretending the same deterministic contract
still applied.

【后来怎么看（截至冻结终点）】

Partly retained. V95 logs later showed that the special one-vector `t+48` Stage
was a bandwidth-poor auxiliary modulator. V96 absorbed the valid semantic
caveat—far future is not the same deterministic action transition—while keeping
all four horizons as spatial evidence in one stage instead of retaining this
special Stage token.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被 V96 吸收；target-granularity principle retained, special Stage token rejected.

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 354-375。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`4/8/12/16`、`4/8/16/24`、`4/12/24`。

【关联】

Long-horizon phases, future-target publicization, later S interval objectives and
Teacher horizon semantics.

### CI-0051 — Causal attention can be bypassed by global summaries

【时间 / 版本】

V95 directed stage/window implementation, 2026-07-21.

【当时在解决什么问题】

Direct attention masks enforced an order between context, action, stage and
window tokens, but a later block's global modulation summary could still average
downstream tokens and feed them back into upstream state.

【当时的 insight / hypothesis】

Causality must be checked across every residual and conditioning path, not only
the obvious attention matrix. A global AdaLN/modulation summary can form a
hidden reverse edge even when attention masks look correct.

【具体做法】

- Updated stage before window within each DiT block.
- Added a dedicated stage-to-window FiLM residual bridge.
- Excluded action, stage and window tokens from the next layer's global
  modulation summary.
- Tested stage-to-window gradient flow and window-to-stage isolation.
- Removed an in-place autograd update discovered by the causal-gradient test.

【观察到的结果】

- Directed tests showed nonzero finite stage-to-window gradients.
- Window perturbation could no longer alter stage through the global modulation
  side path.
- CPU BF16 forward/backward and teacher no-grad boundaries remained finite.

【当时的解释】

The information graph is the union of attention, normalization, modulation,
residual and summary paths. Fixing only attention would have produced a
plausible but false causal contract.

【后来怎么看（截至冻结终点）】

Retained as a source-level closure lesson. Later V95 evidence showed that fixing
the deterministic bypass did not make the special stage carrier sufficient; the
long-horizon one-vector bottleneck was subsequently removed in V96.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；确定性旁路修复正确，原 stage carrier 后来被 V96 替代。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 366-375。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。

【关联】

Forbidden-condition interventions, bypass audits, later G/S/W/P carrier
ownership and bottom ingress.

### CI-0052 — Nonuniform prediction horizons require real action-time boundaries

【时间 / 版本】

V95 temporal alignment repair, 2026-07-21.

【当时在解决什么问题】

After changing future anchors to `4/12/24`, the existing code still split the
24-step action/event timeline into three equal `8/8/8` segments.

【当时的 insight / hypothesis】

Target timestamps and action-conditioning segments must share the same physical
time coordinate. A valid token count does not make an equal partition
semantically correct.

【具体做法】

- Replaced equal thirds with real incremental boundaries `4/8/12` associated
  with cumulative targets `4/12/24`.
- Passed actual horizon offsets rather than ordinal anchor ids.
- Added tests for nonuniform temporal boundaries and cache/non-cache target
  selection.
- Made resume compare effective resolved offsets so an automatically derived
  empty tuple was not falsely incompatible with the same explicit tuple.

【观察到的结果】

The command-line stage weight, resolved horizons, target cache indices and
action/event intervals all reached their intended consumers under tests.

【当时的解释】

The previous equal split would have assigned actions to the wrong future target
while leaving every tensor shape valid, precisely the kind of semantic
misconnection shape tests miss.

【后来怎么看（截至冻结终点）】

Retained. This became another concrete example of why temporal axes must be
source-backed rather than inferred from token counts.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；deterministic temporal misalignment fixed.

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 366-375。
- 段级工件：见 Segment 005 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`4/12/24`、`8/8/8`、`4/8/12`。

【关联】

Flow-time contracts, future intervals, action horizons and later
`4-8/8-16/16-32/32-48` semantics.

## Segment 005 synthesis

The V95 design replaced an early pooled “top” vector with a structured evidence
and prediction graph:

```text
visible DINO patches
  + learned bidirectional patch flow / warp / confidence
        ↓ typed selector/value evidence
directed shared DiT
  clean context → action → stage → window
        ↓
window targets: t+4 / t+12 / t+24
stage target:   coarse t+48 result prior
```

Three corrections inside the segment are as important as the implementation:

1. The assistant initially moved toward implementation before explicit
   permission and was stopped; design discussion continued until the user
   explicitly started the work.
2. Persistent stage memory was proposed first, then rejected in favor of a
   stateless observer that could still read finite visual/action/state history.
3. The `t+48` target was explicitly downgraded from “physical transition” to a
   coarse stage prior because the supplied action ended at `t+24`.

The implementation handoff reported strong static evidence—causal masks,
teacher isolation, BF16, gradients, real horizon alignment and loss-ledger
closure—but no training log yet. At this point the historical record supports
“a non-toy, internally coherent V95 graph was built,” not “Flow-DINO/JEPA or the
stage token improved ClearVLA.”

## Segment 006 — Goal/action conditioning, counterfactual waste and the missing Stage1

### Segment boundary

```text
Date:                 2026-07-21 to 2026-07-22
Visible messages:     376-439
Starts with:          action-history, image-summary and Goal Token proposal
Ends with:            independent V95 representation Stage1 implementation
Primary historical artifacts:
  clearvla/policy/proposal.py
  clearvla/policy/goal_conditioning.py
  clearvla/policy/trunk_primitives.py
  clearvla/policy/system.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  clearvla/cli/train_v40_policy.py
  scripts/current_v95_flow_dino_jepa.sh
  scripts/current_v95_flow_dino_jepa_stage1.sh
  scripts/current_v95_flow_dino_jepa_policy.sh
  attached V95 mixed-training smoke log
```

### CI-0053 — “Action” represented four different lifecycle roles

【时间 / 版本】

V95 mixed-training memory audit, 2026-07-21.

【当时在解决什么问题】

The first V95 training attempt consumed almost the full 24 GB card and failed
while entering a `hold_policy` forward. The user asked why action was present
inside the new future-representation model at all.

【当时的 insight / hypothesis】

Past executed action, noisy candidate action, ground-truth flow-matching action
and artificial counterfactual action are different objects. Action-conditioned
world prediction needs the first three in their proper roles, but does not need
unused hold/shuffle branches.

【具体做法】

- Separated:
  - `executed_history`;
  - noisy action tokens being denoised;
  - `target_action` used to construct the flow target;
  - artificial `hold_action/shuffle_action`.
- Explained that the noisy action is a standard training state, not direct
  leakage of the final answer.
- Traced hold/shuffle to an old contrastive objective intended to prevent the
  future predictor from ignoring action.
- Checked whether any nonzero active loss still consumed those counterfactual
  forwards.

【观察到的结果】

- The active V95 script had disabled the old rollout contrast.
- Hold and shuffle still ran full differentiable policy forwards and retained
  their graphs.
- The main action trajectory and executed history remained necessary to predict
  action-dependent future DINO rather than dataset-average future.

【当时的解释】

The design error was not action conditioning. It was retaining expensive
counterfactual consumers after their objective had left the active experiment.

【后来怎么看（截至冻结终点）】

Retained. This separation later also helped define what an independent
representation Stage1 was and was not allowed to read.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；unused counterfactual actions removed from default V95 training.

【证据与锚点】

- 类型：源码事实；因果干预；过程/决策记录
- 对话：visible messages 383-400。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`hold_policy`、`executed_history`、`target_action`、`hold_action/shuffle_action`。

【关联】

Action-conditioned JEPA, future-target leakage, counterfactual probes and
Stage1/Stage2 boundaries.

### CI-0054 — Future targets and counterfactual forwards need independent demand flags

【时间 / 版本】

V95 first GPU smoke, 2026-07-21.

【当时在解决什么问题】

V95 required future DINO teacher data, so runtime set `use_future=True`.
That same flag also enabled two action counterfactual forwards and zero-weight
layer auxiliary computation.

【当时的 insight / hypothesis】

Producer availability does not imply every historical consumer should run.
Each expensive graph must be materialized from the nonzero active objective or
diagnostic that actually consumes it.

【具体做法】

- Split `need_future_teacher` from `need_action_counterfactuals`.
- Made hold/shuffle run only when their contrastive objective was genuinely
  enabled.
- Skipped layer and midcut auxiliary computation entirely at zero external
  weight.
- Left limited no-grad validation counterfactuals as a possible diagnostic
  rather than a retained train graph.
- Counted policy forwards under contrast off/on.

【观察到的结果】

- Before the fix, batch eight held main, hold and shuffle graphs over a canvas
  that had grown from roughly 332 to 525 tokens and reread about 896 evidence
  tokens.
- Main forward occupied about 22.82 GiB before the next branch requested only
  20 MiB and failed; LayerNorm was merely the allocation site.
- After the repair, contrast-off used one policy forward and explicit contrast
  used three.
- A later batch-two smoke peaked at about 4.69 GiB allocated without per-batch
  growth.

【当时的解释】

The OOM came from graph multiplicity and quadratic canvas growth, not from the
4.36M flow module, BF16, LayerNorm or a tiny final allocation.

【后来怎么看（截至冻结终点）】

Retained as a demand-driven computation rule. It also exposed that a zero loss
weight is not enough if the producer graph is still executed for logging.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；deterministic compute waste fixed.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 383-400 and 411-417。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`need_future_teacher`、`need_action_counterfactuals`。

【关联】

Loss ownership, zero-weight semantics, memory gates and later Teacher isolation.

### CI-0055 — Long action history should preserve recent detail and compress distant context

【时间 / 版本】

V95 action-history extension, 2026-07-21.

【当时在解决什么问题】

The stateless stage observer initially saw only three executed actions at
offsets `-8/-4/-1`, which was sparse evidence for progress through a task.

【当时的 insight / hypothesis】

History length should not mean a flat set of interchangeable action tokens.
Recent actions need precise identity; distant actions can be summarized while
retaining real relative time and changes.

【具体做法】

- Expanded history offsets to
  `-24,-16,-12,-8,-6,-4,-2,-1`.
- Compressed the distant portion into three summary tokens and kept four recent
  actions individually.
- Added continuous relative-time encoding, action deltas and distinct
  recent/summary roles.
- Reused one time-aware history encoding rather than separately recompiling the
  same actions for proposal and canvas.
- Deferred sparse image-summary history until cache/runtime semantics could be
  made identical.

【观察到的结果】

- Eight historical actions produced seven condition tokens.
- CPU BF16 forward/backward gave the action-history encoder finite gradients.
- Later GPU smoke reported action-history gradient around
  `6.5e-2` to `1.1e-1`.

【当时的解释】

This provided more phase evidence without adding recurrent hidden state or
allowing a long flat history to dominate current visual evidence.

【后来怎么看（截至冻结终点）】

Implemented and gradient-connected. No later matched ablation in the frozen
archive isolated the contribution of this exact history compressor, while the
recent-detail/distant-summary principle survived later history interfaces.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；history-compression 原则保留，孤立收益未被验证。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；历史检索；推断/未决
- 对话：visible messages 376-405 and 411-417。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`-8/-4/-1`。

【关联】

CI-0049, stateless phase inference, history compression and later language/history
conditioning.

### CI-0056 — Goal and action may share a condition space without sharing identity

【时间 / 版本】

V95 Goal Token design, 2026-07-21.

【当时在解决什么问题】

The existing `task_token` was only learned constant state. The user wanted a
small number of language-derived Goal Tokens located near action conditioning,
but not numerically or semantically equated with action.

【当时的 insight / hypothesis】

Goal says what should be achieved; action history says what has already been
done. They can enter a shared hidden width and downstream mixer while retaining
private stems, role embeddings and source identity.

【具体做法】

- Added a query resampler producing four Goal Tokens.
- Put Goal and action-history tokens adjacent on the condition canvas.
- Used separate normalization/projection stems followed by a shared condition
  mixer.
- Let both reach dynamics/stage/window and the final action consumer.
- Avoided an explicit Goal-action cosine alignment objective.
- Added Goal/action norms, pair cosine, cross cosine and gradient diagnostics.
- Added the Goal Resampler to an optimizer group after static review found it
  had gradients but initially no optimizer owner.

【观察到的结果】

- GPU smoke showed finite nonzero Goal gradients around
  `1.0e-2` to `1.5e-2`.
- Goal and action-memory norms were both near `4.2`, while their cosine stayed
  low to moderate rather than becoming identical.
- Four Goal Tokens had pair cosine `0.969-0.977`, indicating strong
  redundancy.

【当时的解释】

The redundancy was expected if the supplied T5 condition had only one valid
pooled token: multiple queries can only attend to the same value. Gradient
existence therefore proved connectivity but not language-role diversity.

【后来怎么看（截至冻结终点）】

Partly retained. The condition-space ownership was implemented, but the
single-instruction task and pooled-token realizations never supplied evidence
for multi-goal language semantics. Later weak-language findings preserved that
limitation rather than resolving it.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；condition-space 分权保留，多目标语言有效性未被当前数据识别。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 376-417。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`task_token`。

【关联】

Goal token diversity, task conditioning, optimizer ownership and later language
publicization.

### CI-0057 — Precomputed language embeddings should be content-addressed inputs

【时间 / 版本】

V95 T5 adaptation, 2026-07-21.

【当时在解决什么问题】

The user already had a task-condition `.pt` and did not want tokenizer/T5
loading, networking or model-cache state inside the training process.

【当时的 insight / hypothesis】

Language encoding can be an immutable experiment input. The active network
should train only the small resampler, validate the tensor/mask contract and
fingerprint the embedding content rather than its filename.

【具体做法】

- Removed runtime T5/tokenizer/transformers loading from the formal path.
- Accepted `[L,D]`, `[1,L,D]` and common dict fields with optional masks.
- Added `--t5-condition-path` / `T5_CONDITION_PATH`.
- Validated dimensions, valid token count and content hash.
- Required Stage1/Stage2 and resume to use the same embedding fingerprint.

【观察到的结果】

- A repository `[1,4096]` RDT T5 tensor completed forward/backward with
  Goal-Resampler gradients.
- The implementation distinguished interface testing with an empty/pooled
  condition from meaningful multi-token language conditioning.

【当时的解释】

Content addressing prevented two different language conditions with similar
filenames from sharing an incompatible checkpoint and kept the large language
encoder outside GPU/runtime ownership.

【后来怎么看（截至冻结终点）】

Retained. Multi-task datasets would still require per-sample language conditions
rather than one fixed file for an entire run.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；input/checkpoint contract implemented.

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 401-417 and 428-439。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`T5_CONDITION_PATH`。

【关联】

Run manifest, Goal Tokens, task identity and later normalizer/source fingerprints.

### CI-0058 — Process allocation and device-context usage are different memory metrics

【时间 / 版本】

V95 batch-two GPU smoke, 2026-07-21.

【当时在解决什么问题】

The log reported roughly `ctx_used=17.626/23.559 GiB`, which could be read as
the repaired V95 process still consuming about 18 GB.

【当时的 insight / hypothesis】

PyTorch allocated/reserved/peak values describe the process allocator; device
`total-free` includes other processes and CUDA context. They must not be
reported as the same model footprint.

【具体做法】

- Compared after-model, stable train, peak train and validation allocator
  readings.
- Subtracted process reserved memory from total device used as a rough
  attribution check.
- Checked batch-to-batch stability for leaks.

【观察到的结果】

- Model-on-device allocation was about `0.611 GiB`.
- Stable training allocation was about `2.75 GiB`, peak allocation
  `4.692 GiB`, reserved `4.783 GiB`, and validation peak about
  `3.34 GiB`.
- Roughly `12.8 GiB` of the device-used figure came from outside this
  PyTorch allocator.
- Memory remained stable after batch two.

【当时的解释】

The counterfactual fix had materially reduced the active graph. The card could
still be unsafe for batch eight because of other occupants, but that was not the
same claim as V95 requiring 18 GB.

【后来怎么看（截至冻结终点）】

Retained as a telemetry rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 406-417。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`ctx_used=17.626/23.559 GiB`。

【关联】

CUDA memory logging, batch-size release gates and shared-GPU attribution.

### CI-0059 — Model stage tokens and training Stage1 are unrelated axes

【时间 / 版本】

V95 experiment-orchestration correction, 2026-07-21 to 2026-07-22.

【当时在解决什么问题】

The assistant described V95 as a single policy-stage run initialized from an old
Stage1 checkpoint, while the user had explicitly intended a new Stage1 for the
new top representation.

【当时的 insight / hypothesis】

`stage_h=48` inside the model, an old Stage1 initialization checkpoint and an
independent representation-training Stage1 are three different concepts. A new
network integrated into policy training is not a replacement for the requested
new representation experiment.

【具体做法】

- Traced the original V95 wrapper to
  `--training-stage policy`.
- Confirmed there was no new V95 Stage1 script, loss, validation criterion or
  representation checkpoint.
- Stopped expanding the mixed-training smoke.
- Reopened Stage1 producer, optimizer, validation, save and Stage2-consumer
  contracts.

【观察到的结果】

- The new Flow-DINO, stage, Goal and history modules had been inserted into the
  old policy run and jointly cold-started.
- The old `best_contract.pt` still acted as initialization.
- The first answer wrongly rationalized this as the previously discussed
  end-to-end attachment path; the user corrected that interpretation.

【当时的解释】

This was not a naming issue or a missing shell alias. The requested experiment
boundary and its learning objective had not been implemented.

【后来怎么看（截至冻结终点）】

Corrected in the same segment through explicit Stage1 and policy entrypoints,
dedicated losses and a typed representation checkpoint.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留 as a negative lesson; implementation omission later repaired.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 418-439。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`best_contract.pt`。

【关联】

Training lifecycle identity, checkpoint ABI, top representation pretraining and
later mainline-stage confusion.

### CI-0060 — Representation Stage1 must not see the future target action

【时间 / 版本】

V95 dedicated Stage1 repair, 2026-07-22.

【当时在解决什么问题】

After Stage1-specific loss was added, it still called the old
`flow_training_forward`. That path noised the ground-truth future action and
inserted it as trajectory tokens before computing representation targets.

【当时的 insight / hypothesis】

Removing action loss is not enough to create action-free representation
pretraining. If future target action enters the online predictor, future
representation loss can exploit label-side action information and no longer
tests inference from current evidence/history/goal.

【具体做法】

- Added a dedicated `flow_jepa_stage1_forward` whose interface has no future
  target action.
- Allowed current visual/state, executed-action history and T5 Goal Tokens.
- Kept future visual exclusively in the frozen/no-grad DINO teacher.
- Skipped final action decoder, execution controller and old layer contracts.
- Used only window/stage JEPA and flow geometry losses.
- Counted action-decoder calls and asserted zero.

【观察到的结果】

- CPU forward/backward produced gradients in Flow-DINO, DiT, Goal Resampler and
  action-history encoder.
- Final action decoder was called zero times and layer-contract output was empty.
- Stage1 checkpoint contract recorded the no-target-action/no-action-decoder
  invariant.

【当时的解释】

The old forward had formed a subtle teacher-forcing shortcut while every shape
and representation loss looked valid. A dedicated producer boundary was needed,
not just different loss weights.

【后来怎么看（截至冻结终点）】

The no-future-leakage rule is retained. The independent Stage1 was retired into
V96's single-stage joint training before it became a long-run donor design, so
absence of leakage—not independent Stage1 quality—is the durable result.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；泄漏修复保留，独立 Stage1 realization 已退役。

【证据与锚点】

- 类型：日志观测；测试/运行观测；方案/设计记录；推断/未决
- 对话：visible messages 423-439。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`flow_training_forward`、`flow_jepa_stage1_forward`。

【关联】

CI-0045, target leakage, action-conditioned Stage2 and later Teacher isolation.

### CI-0061 — A new Stage1 needs its own objective, optimizer, validation and checkpoint ABI

【时间 / 版本】

V95 independent Stage1 closure, 2026-07-22.

【当时在解决什么问题】

The repository's pre-existing `training_stage=stage1` optimized old
layer-contract/rollout probes, selected checkpoints with old validation and used
optimizer groups unrelated to the new Flow-JEPA computation.

【当时的 insight / hypothesis】

A training stage is closed only when its forward, losses, trainable owners,
validation statistic, checkpoint selection, serialized identity, logger and
downstream loader all describe the same graph.

【具体做法】

- Defined a Flow-DINO/JEPA representation objective instead of old contract
  losses.
- Built a dedicated optimizer from the actual Stage1 graph.
- Added `final_norm`, which carried the prediction path but was initially
  omitted, and removed inactive contract/action/execution parameters.
- Added the invariant that every parameter with nonzero backward gradient has an
  optimizer owner.
- Selected `best_stage1_representation.pt` by representation validation rather
  than action RMSE.
- Required Stage2 to validate Flow-JEPA architecture, Goal dimension, T5 hash,
  source hashes and the no-action-leak contract.
- Made old `best_contract.pt` invalid for the new policy Stage2.
- Added separate Stage1 and policy wrapper scripts.
- Extended the log audit tool to recognize both `[v95-stage1-*]` and
  `[v95-*]`.
- Fixed a no-grad validation false failure in a training-only
  `requires_grad` assertion.

【观察到的结果】

- Sixty-six CPU forward/backward, BF16, controller, logging and checkpoint
  contract checks passed.
- Stage1 validation no longer computed meaningless action RMSE.
- Stage2 refused to start without the explicit compatible Stage1 checkpoint.
- Real GPU/dataset smoke was still pending.

【当时的解释】

The successive defects—wrong forward, wrong optimizer owner, wrong validation,
wrong logger and permissive loader—were all manifestations of a half-created
stage. Fixing only the shell entry would have preserved most of them.

【后来怎么看（截至冻结终点）】

Retained as the segment's central closure standard. The particular independent
Stage1 was later absorbed by V96's single-stage end-to-end design, so its static
closure was not promoted into a long-run architectural success.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；lifecycle principle retained, independent Stage1 retired.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；方案/设计记录；推断/未决
- 对话：visible messages 423-439。
- 段级工件：见 Segment 006 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`final_norm`、`best_stage1_representation.pt`、`best_contract.pt`、`requires_grad`。

【关联】

Mandatory subsystem familiarity, optimizer ownership, Stage1/Stage2 checkpoint
ABI and later schema lifecycle contracts.

## Segment 006 synthesis

This segment exposed two different kinds of accidental coupling:

```text
future teacher required
    was incorrectly treated as
hold/shuffle counterfactual graphs required

new representation modules exist in policy training
    was incorrectly treated as
new representation Stage1 exists
```

The first coupling caused a direct memory failure; the second caused a more
dangerous experimental-semantic failure. Removing hold/shuffle reduced the
observed batch-two process peak to about 4.7 GiB and made Goal/action-history
conditioning measurable. But the user's Stage1 correction forced a deeper
review that found an action-label leak, missing optimizer ownership, old
validation semantics and permissive checkpoint reuse.

By visible message 439, the independent Stage1 had a dedicated action-free
online forward, future-only teacher, representation losses, optimizer, validation
criterion, checkpoint identity, logger and strict Stage2 consumer. The record
therefore includes both the initial failure and the repaired closure; it does
not yet include a real GPU Stage1 result.

## Segment 007 — Mixed-training evidence, flow baselines and the V96 late bottleneck

### Segment boundary

```text
Date:                 2026-07-22
Visible messages:     440-539
Starts with:          audit of the completed old V95 mixed-training run
Ends with:            V96 single-stage late-bottleneck implementation handoff
Primary historical artifacts:
  completed V95 mixed-policy nohup log
  Flow-DINO dataset-motion probe and its shell wrapper
  clearvla/policy/flow_dino_evidence.py
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/trunk.py
  clearvla/policy/config.py
  scripts/current_v96_late_bottleneck_jepa.sh
```

### CI-0062 — V95 mixed training was strongest early and then optimized the easy horizon

【时间 / 版本】

Completed old-style V95 mixed-policy run, 2026-07-22.

【当时在解决什么问题】

A long V95 run reached low aggregate RMSE, but it predated the independent
Stage1 contracts and jointly optimized action, Flow-JEPA and execution from an
old contract checkpoint.

【当时的 insight / hypothesis】

Joint training can reveal genuine structural potential and still choose the
wrong late-training direction. Dense direct action supervision may continue
improving the first steps after representation and long-horizon gains have
stalled.

【具体做法】

- Identified the run as `training_stage=policy` initialized from old
  `best_contract.pt`, with hundreds of missing/mismatched keys.
- Compared it with the healthiest V94 at aligned epoch two.
- Followed absolute first, tail, arm, gripper, balanced score, deploy gate,
  JEPA validation and controller ablations through 16 epochs.
- Distinguished best aggregate RMSE from best deployment/balanced checkpoint.

【观察到的结果】

- At epoch two V95 beat V94: action RMSE `0.08791` versus `0.09654`,
  gripper `0.15139` versus `0.17629`, and tail/first `2.504` versus
  `3.876`.
- From epoch two to the end, first RMSE improved about 65%, while tail improved
  only about 8.5%; tail/first rose to `6.538`.
- Aggregate RMSE reached about `0.0783`, but epoch two was the only checkpoint
  passing the deployment gate and had the best balanced score.
- Window validation improved modestly; long-range stage validation stopped
  improving around epoch six and retained a much larger train/validation gap.
- Goal/action history and value-reader gradients were live, but Goal tokens
  converged to near-common direction.

【当时的解释】

The new top path was not useless: its early aligned result was better than V94.
Later optimization disproportionately exploited the short direct action path,
so selecting by average RMSE discarded a healthier earlier balance.

【后来怎么看（截至冻结终点）】

Retained as evidence against both extremes: V95 was neither a total failure nor
a validated final architecture. It motivated a single-stage redesign with
better long-horizon ownership rather than simply more epochs.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；mixed run archived as an evolutionary baseline, not current-proof.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 440-451 and 480-491。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`best_contract.pt`。

【关联】

Checkpoint selection, horizon imbalance, early stopping and later eight-epoch
recovery gates.

### CI-0063 — Small flow magnitude requires a data-normalized zero-flow baseline

【时间 / 版本】

V95 flow audit and dataset probe, 2026-07-22.

【当时在解决什么问题】

The mixed log showed patch-flow magnitude shrinking from roughly `0.196` to
`0.017`, cycle loss falling by two orders of magnitude and warp loss changing
little. It was initially labelled a near-zero-flow shortcut.

【当时的 insight / hypothesis】

Magnitude alone cannot distinguish collapse from correct small motion. Units,
temporal interval, spatial averaging, static-background share and a matched
zero-flow warp baseline are required.

【具体做法】

- Corrected the initial claim after the user noted that frames genuinely changed
  little.
- Established that learned flow compared `-8→-4` and `-4→0`, four-frame
  intervals on an `8x8` patch grid.
- Designed and ran a data-only DINO correspondence probe over
  `1/4/12/24/48` horizons.
- Reported identity MSE, local-oracle gain, nonzero displacement, mutual match,
  motion-top statistics and soft-correlation displacement.
- Aligned the probe's MSE with the training logger's mean-root-error semantics.

【观察到的结果】

- Five validation episodes and 1432 valid centers yielded 183,296 patch pairs
  per horizon.
- At four frames, about 16.3% of patches had nonzero local best match and the
  motion-top 20% had 64.1%; local oracle gain was about 14.1%.
- At 48 frames, nonzero matching rose to 43.6%, motion-top nonzero to 91.6% and
  mutual matches fell to 59.4%.
- A mixture of roughly 20% moving patches at 0.08 patch average displacement
  yields a global mean near `0.016`, compatible with `0.017`.
- Untrained global soft DINO correlation was worse than identity at every
  horizon and produced exaggerated fuzzy displacement.

【当时的解释】

Motion was sparse, local and sub-patch at short horizons but accumulated
substantially over time. The learned model may have been correcting an initially
false broad correspondence rather than collapsing valid motion.

【后来怎么看（截至冻结终点）】

The “zero-flow collapse” claim was downgraded to essentially excluded by the
available evidence. The remaining correct test was learned versus matched
zero/shuffled warp gain, especially on motion patches.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留 as a corrected hypothesis; the original strong collapse claim was withdrawn.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；数据探针；过程/决策记录
- 对话：visible messages 452-479 and detailed retrospective 5488-5490。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1/4/12/24/48`。

【关联】

Data priors, intervention baselines, spatial units and later flow-address design.

### CI-0064 — A legal negative uncertainty NLL can corrupt representation scoring

【时间 / 版本】

V95 mixed-training objective audit, 2026-07-22.

【当时在解决什么问题】

The uncertainty likelihood term became strongly negative and offset most of the
positive future/stage prediction contributions in the representation group and
checkpoint score.

【当时的 insight / hypothesis】

A likelihood term can legitimately be negative after constants are omitted, but
that does not make it an appropriate additive progress score for representation
quality or checkpoint selection.

【具体做法】

- Reconstructed uncertainty as approximately
  `detached_error / scale + log(scale)`.
- Separated its training contribution from window/stage prediction metrics.
- Compared the positive future/stage budget before and after NLL cancellation.
- Flagged Stage1 selection by aggregate validation loss as vulnerable to
  increasingly negative uncertainty.

【观察到的结果】

- Late positive future/stage contributions totalled roughly `0.019`, while
  uncertainty contributed about `-0.0155`, leaving a deceptively small
  representation group near `0.004`.
- Stage validation itself had largely stopped improving.

【当时的解释】

The objective remained mathematically finite, but the scalar no longer meant
“future representation error.” Training could retain calibrated NLL while
checkpoint ranking needed a non-cancellable representation measure.

【后来怎么看（截至冻结终点）】

Retained as a confirmed V95 scoring defect. V96 replaced the affected future
organization; the archive contains no evidence that this exact NLL should be
recovered or repaired as an independent current mechanism.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；负 NLL 的审计教训保留，原 scoring path 已被替代。

【证据与锚点】

- 类型：日志观测；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 440-451 and 480-491。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`detached_error / scale + log(scale)`。

【关联】

Loss ledgers, checkpoint selection, uncertainty calibration and later Teacher
diagnostics.

### CI-0065 — The special global Stage token was a long-horizon bottleneck

【时间 / 版本】

V95 long-horizon audit, 2026-07-22.

【当时在解决什么问题】

V95 represented `t+4/12/24` with dense spatial window tokens but compressed
`t+48` across space/cameras into one Stage token that reached the action tower
only through a scalar-gated Stage-to-window bridge.

【当时的 insight / hypothesis】

Long-range information had supervision but little representational bandwidth or
decision ownership. A stronger loss on the same single token would not restore
lost object/space structure.

【具体做法】

- Traced the Stage token's absence from direct Evidence-MMDiT inputs.
- Measured its only route through Stage-to-window affine modulation.
- Compared growing update norm with falling Stage/window cosine and stagnant
  tail benefit.
- Compared dense short targets with a single coarse long target and a 24-step
  action horizon.

【观察到的结果】

- Stage validation plateaued, train/validation differed by roughly `2.35x`,
  and Stage-to-window update grew from about `0.26` to `2.16`.
- Stage/window cosine fell toward `0.06-0.08`.
- The action system continued to improve first steps while tail barely moved.
- Independent Stage1's neutral future action also mismatched the policy-time
  action-conditioned future contract.

【当时的解释】

Long-range state was an auxiliary modulation rather than a first-class evidence
source. Its one-vector carrier could not preserve localized motion that the data
probe showed increasing with horizon.

【后来怎么看（截至冻结终点）】

V96 removed the special Stage token and kept all `4/12/24/48` horizons as
spatial evidence in one joint training path.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；identified bottleneck removed in V96.

【证据与锚点】

- 类型：源码事实；日志观测；数据探针
- 对话：visible messages 480-522 and detailed retrospective 5488-5490。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`t+4/12/24`、`4/12/24/48`。

【关联】

CI-0050, long-horizon publicization, later interval identity and future-object
dynamics.

### CI-0066 — Flow has both a local-address role and a world-model role

【时间 / 版本】

V95-to-V96 design correction, 2026-07-22.

【当时在解决什么问题】

A proposed unified long-horizon JEPA risked demoting Flow to a generic
conditioning feature and losing its direct high-precision effect on action
refinement.

【当时的 insight / hypothesis】

Visual patch flow and action flow matching are distinct. Patch flow should give
local spatial address, confidence, occlusion and warp evidence directly to
action queries, while DiT uses that evidence for longer-horizon world prediction.

【具体做法】

- Preserved direct Flow-DINO selector/value evidence into the final action
  MMDiT.
- Kept action-flow velocity prediction as the single action generator.
- Prevented long-horizon tokens from replacing the local flow/raw/DINO path.
- Treated patch Flow as observed motion evidence rather than a linear
  `t+48` extrapolator.

【观察到的结果】

Source review confirmed the existing action blocks already reread Flow content,
motion and warp evidence each denoising step, while long-range Stage had only an
indirect bridge.

【当时的解释】

Long-range semantics and fine action geometry are complementary. Making either
one the sole bottleneck would repeat the project's pattern of solving one axis
by deleting another.

【后来怎么看（截至冻结终点）】

Retained in V96's coarse/global plus late local-address reader.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；architectural role clarified and implemented.

【证据与锚点】

- 类型：源码事实；方案/设计记录
- 对话：visible messages 487-522。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。

【关联】

Two meanings of flow, late precision reading and later raw-image grounding.

### CI-0067 — The lower stack had been renamed and reorganized, not removed

【时间 / 版本】

V95 top-to-bottom source archaeology, 2026-07-22.

【当时在解决什么问题】

The user asked how the new top reached the old CVAE, hierarchical MMDiT and
workspace, and whether those components had been dismantled.

【当时的 insight / hypothesis】

Class names are a poor guide to functional ancestry. Random CVAE state and
stateful workspace control were mostly gone, while deterministic latent
organization, typed evidence memory, layer hierarchy and adaptive action MMDiT
remained substantial.

【具体做法】

- Traced stochastic posterior, `mu/logvar`, reparameterization and KL paths.
- Traced eight top DiT blocks, layer contracts, consequence, controlled
  dynamics and the three-block Evidence-MMDiT decoder.
- Traced workspace controller/slots separately from `OwnedEvidenceMemoryBank`.
- Verified route, dwell, capacity, terminal and value-reader switches in the
  active wrapper and log.

【观察到的结果】

- CVAE stochastic core was absent; `z` was deterministic, while many
  `latent_cvae_*` names remained.
- The old hierarchical decoder class was absent, but its function was replaced
  by eight top DiT blocks plus three adaptive action MMDiT blocks.
- Stateful workspace slots/controller/promotion were off; typed evidence memory
  remained as the central storage/read interface.
- V95 logs showed the adaptive execution path actually ran.

【当时的解释】

The system contained two substantial towers plus an intermediate hierarchy, not
a clean new top over a minimal bottom. Future changes needed to preserve useful
adaptive execution while reducing duplicated carrier semantics.

【后来怎么看（截至冻结终点）】

Retained. V96 explicitly preserved the lower execution stack rather than
silently deleting it.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；历史检索
- 对话：visible messages 498-510。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`mu/logvar`。

【关联】

Architecture ancestry, renamed modules, later V120 replay and bottom ABI.

### CI-0068 — Flow-augmented evidence was not yet a Flow-conditioned reader

【时间 / 版本】

V95 reader audit, 2026-07-22.

【当时在解决什么问题】

Current code exposed Flow-DINO selector/value tokens, but the user asked whether
Flow could act as an actual spatial address prior for conditional DINO reading.

【当时的 insight / hypothesis】

Encoding displacement into an attention key is not equivalent to using
displacement to define where values are sampled. A true conditional reader needs
sample- and action-timestep-specific spatial priors, with DINO/raw features as
values and confidence/uncertainty controlling address distribution.

【具体做法】

- Traced content/motion/warp selector-value pairs into ordinary dot-product
  attention.
- Found Flow-DINO `key_bias` was zero and the supported bias shape was static
  across batch/action time.
- Proposed `grid + flow + action-conditioned offset` local sampling with
  confidence/occlusion weighting and identity fallback.
- Immediately clarified to the user that this explicit deformable reader was a
  proposed extension, not already implemented history.

【观察到的结果】

Existing code let Flow features influence learned selection but did not make
Flow coordinates directly select DINO positions.

【当时的解释】

The prior design was accurately renamed “Flow-augmented DINO evidence.” The
proposed reader would make spatial ownership explicit without hard top-k or a
second action tower.

【后来怎么看（截至冻结终点）】

Implemented in V96 as a late source-query/target-candidate local reader, with
soft importance and identity fallback.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；proposal/source distinction recorded, extension later implemented.

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；方案/设计记录
- 对话：visible messages 498-539。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`key_bias`。

【关联】

Selector/value ownership, deformable reading, raw-image detail and P1 precision.

### CI-0069 — Precision should increase after importance is known, not before information exists

【时间 / 版本】

V96 late-bottleneck design, 2026-07-22.

【当时在解决什么问题】

Running native `24x24` all-pairs correlation everywhere would raise compute
dramatically, while compressing all visual content to `8x8` before deciding
what mattered would erase local detail.

【当时的 insight / hypothesis】

Use coarse global organization to establish motion/JEPA importance, then spend
fine local correlation and reading bandwidth near relevant regions. Preserve a
coarse semantic/identity fallback so an imperfect importance map cannot delete
information.

【具体做法】

- Kept global all-pairs Flow on `8x8`.
- Combined motion, confidence, occlusion/entropy and JEPA read importance into a
  soft fine-detail gate.
- Distinguished JEPA mask (“what is hidden/predicted”) from action importance.
- Ran native-grid local refinement and address candidates without hard top-k.
- Used source-frame queries and target-frame flow/identity neighborhoods with
  explicit out-of-bounds masks.
- Preserved four spatial horizons `4/12/24/48`; only the first three map to
  the 24-step action segments.
- Added a future-change constraint beside absolute future semantics to reduce
  copy-current shortcuts.

【观察到的结果】

- Per frame pair, rough accounting was about 4096 coarse comparisons, 1600 fine
  local candidates and 1152 address reads versus 331,776 native all-pairs.
- Production BF16 geometry retained `24x24` DINO input, `8x8` coarse chart,
  896 evidence tokens and 512 future tokens.
- The top encoder had about 3.13M trainable parameters and passed 75 related
  tests.
- Soft weighting reduced effective influence but did not skip fixed candidate
  kernels; logging separated weighted from actually computed comparisons.

【当时的解释】

The late bottleneck enlarged usable bandwidth without pretending soft gates
saved FLOPs. Identity/coarse fallback prevented an early mask/flow error from
making a fine region permanently inaccessible.

【后来怎么看（截至冻结终点）】

Implemented as V96, but still used DINO rather than raw RGB; the very next
segment reopened whether genuinely fine contact detail had ever entered the
network.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；V96 static contract implemented, raw-detail adequacy unresolved.

【证据与锚点】

- 类型：测试/运行观测；推断/未决
- 对话：visible messages 511-539。
- 段级工件：见 Segment 007 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`4/12/24/48`。

【关联】

Late information bottlenecks, conditional sampling, computational truth and
later V97 raw pyramid.

## Segment 007 synthesis

This segment is a model of evidence-driven reversal:

```text
log: flow magnitude became very small
initial hypothesis: near-zero-flow shortcut
user objection: the frames genuinely change very little
unit/source audit: four-frame, 8x8, all-patch mean
data probe: sparse sub-patch short motion, accumulating long motion
revised conclusion: small mean flow is plausible; global soft matching was the
                    more suspicious prior
```

The completed mixed run likewise resisted a one-line verdict. V95 was already
better than V94 around epoch two, yet later joint training over-optimized the
first steps, weakened balance and left stage/execution semantics unproven.

V96 responded by removing the special global Stage token and returning to one
joint path with four spatial horizons, explicit future-change supervision and a
late Flow-conditioned address reader. It preserved the old adaptive lower
solver and reported honest fixed candidate computation. At the segment boundary
the new graph was statically coherent, but its “fine” content was still native
DINO rather than raw image detail—exactly the issue opened by visible message
540.

## Segment 008 — V97 role specialization, raw-flow failure and V98 correction

### Segment boundary

```text
Date:                 2026-07-22
Visible messages:     540-604
Starts with:          challenge to the homogeneous eight-block V96 trunk and its DINO-only "fine" reader
Ends with:            V96/V97 full-log audit, zero-flow numerical repair and the V98 corrective contract
Primary historical artifacts:
  clearvla/policy/flow_dino_evidence.py
  clearvla/policy/trunk.py
  clearvla/policy/trunk_primitives.py
  clearvla/policy/time_domain_mmdit.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  clearvla/cli/train_v40_policy.py
  scripts/current_v96_late_bottleneck_jepa.sh
  scripts/current_v97_raw_flow_332_jepa.sh
  scripts/current_v98_dino_seeded_raw_flow_332_jepa.sh
  attached V96/V97 early and extended logs
```

### CI-0070 — A block role exists only when read/write authority and supervision differ

【时间 / 版本】

V96-to-V97 architecture review, 2026-07-22.

【当时在解决什么问题】

V96 described an eight-block top hierarchy, but the user questioned whether the
blocks actually performed different jobs or merely repeated the same denoising
operation under different role names.

【当时的 insight / hypothesis】

Layer embeddings, role labels and auxiliary contracts do not create a hierarchy.
A real role boundary needs different source visibility, token write authority,
downstream ownership and supervision placement while preserving a continuous
residual/gradient path.

【具体做法】

- Traced all eight instances of `TemporalDynamicsBoundDiTBlock`.
- Compared their self-attention, visual cross-attention, rollout cross-attention
  and FFN operations.
- Checked the active high/low-layer weights and found both set to `1.0`.
- Distinguished the eight network-depth blocks from the five flow-sampling
  evaluations and from the separate three-block Evidence-MMDiT decoder.
- Replaced label-only roles in V97 with grounding/world/policy source and write
  permissions.

【观察到的结果】

The V96 blocks were operationally homogeneous; role information was largely a
side contract. In the V97 implementation, the `3+3+2` groups received different
write permissions and their gradients could be logged independently.

【当时的解释】

The earlier hierarchy was descriptive rather than causal. Specialization did
not require eight unrelated block classes, but it did require the graph to make
some writes legal in one group and illegal in another.

【后来怎么看（截至冻结终点）】

The `3+3+2` boundary survived the immediate V97 failure analysis: the failure
was localized to raw-flow construction rather than evidence that role
specialization itself was dead. Later architectures retained explicit role
ownership but replaced the V97 realization, whose long-run usefulness was not
established.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；role-ownership 原则保留，V97 realization 已被替代。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 540-555 and 560。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

CI-0052, CI-0067, role ownership, repeated blocks versus repeated solver calls.

### CI-0071 — Native DINO detail is not raw-image detail

【时间 / 版本】

V96 input-path audit, 2026-07-22.

【当时在解决什么问题】

V96 called its late `24x24` DINO read high precision, but the user asked where
raw images had gone and whether small contact details had again been compressed
before entering the model.

【当时的 insight / hypothesis】

Higher-resolution reading within a semantic feature grid is not equivalent to
access to pixel-level boundaries, texture and contact geometry. DINO and raw RGB
have complementary ownership and must remain explicit, typed inputs rather than
being silently conflated.

【具体做法】

- Traced `condition_mode=dinov2-cache` through the dataset and model manifest.
- Verified that V96 did not return `history_obs_image` and declared
  `flow_jepa_rgb_backbone=False`.
- Identified the actual V96 fine path as a local read over DINO patches.
- Introduced separate `visual` and `raw_visual` batch/model interfaces in V97.
- Kept cached DINO for semantics and future supervision while loading raw RGB
  only for observed history frames.

【观察到的结果】

V96 had no online raw-image reader. V97 added raw history without loading future
RGB or an online full DINO encoder.

【当时的解释】

The distinction prevented an implementation from satisfying “raw detail” by
renaming a denser DINO read. It also made raw activation cost independently
auditable.

【后来怎么看（截至冻结终点）】

Retained. The first raw implementation was faulty, but the DINO/raw separation
was not rejected; V98 repaired their ordering and ownership.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 540-550, 560 and 576-579。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`history_obs_image`、`raw_visual`。

【关联】

CI-0068, CI-0069, information provenance and raw-branch memory accounting.

### CI-0072 — Importance cannot recover detail that was destroyed before importance was known

【时间 / 版本】

V97 raw-pyramid and block-partition design, 2026-07-22.

【当时在解决什么问题】

An initial proposal would organize a coarse map first and consult raw imagery
only near locations already judged important. The user objected that small
contact regions could disappear before that judgment.

【当时的 insight / hypothesis】

Late conditional computation is safe only if an information-preserving memory
still exists at the late read boundary. A coarse selector cannot rediscover a
feature that the only value path has already pooled away.

【具体做法】

- Compared `3+4+1` with `3+3+2`; rejected a single-block world-to-policy bridge.
- Built raw features at `84x84 / 42x42 / 21x21` from `336x336` history frames.
- Kept the `84x84` memory through all three grounding blocks.
- Delayed fixed-size deformable evidence materialization until the grounding
  boundary.
- Kept the existing three-block action decoder downstream rather than adding a
  second raw action head.

【观察到的结果】

V97 could expose `84x84` features to grounding while emitting a compact evidence
interface after block three. This did not by itself prove that the learned flow
selected those details correctly.

【当时的解释】

The design moved the bottleneck after evidence organization without letting the
high-resolution carrier propagate through the whole world/policy stack.

【后来怎么看（截至冻结终点）】

Partially retained in V98. High-resolution detail remained until grounding, but
the raw global-matching stage was removed after logs showed it was the wrong way
to establish the address.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；information boundary retained, initial address mechanism replaced.

【证据与锚点】

- 类型：日志观测；过程/决策记录；推断/未决
- 对话：visible messages 543-550, 553 and 583-594。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`84x84 / 42x42 / 21x21`。

【关联】

CI-0069, late bottlenecks, high-resolution side memory and fixed evidence ABI.

### CI-0073 — Flow should bridge semantic correspondence to physical detail

【时间 / 版本】

V97 design intent, 2026-07-22.

【当时在解决什么问题】

The user argued that Flow had little value if it only estimated displacement
between already-compressed DINO tokens and never addressed real image content.

【当时的 insight / hypothesis】

In this architecture, Flow is useful as a continuous coordinate bridge:
semantic correspondence establishes coarse identity, observed raw motion
refines physical location, and the resulting field addresses high-resolution
values for world and action consumers.

【具体做法】

- Assigned DINO semantics, raw appearance/geometry and motion separate roles.
- Built coarse-to-fine raw flow and a differentiable address reader.
- Preserved soft continuous sampling with identity fallback rather than hard
  top-k regions.
- Routed grounded evidence through world blocks and policy blocks into the one
  existing action decoder.

【观察到的结果】

The V97 graph was connected: raw pyramid, high-flow and address-reader gradients
were non-zero, and policy updates reached the decoder. Later logs showed that the
actual coarse matching did not implement the intended DINO-seeded bridge.

【当时的解释】

Graph reachability established that the idea could participate in training, but
not that the implementation embodied the intended correspondence semantics.

【后来怎么看（截至冻结终点）】

The intent was retained and made more literal in V98: DINO supplied the global
address and raw features supplied bounded local refinement. Later matched
interventions showed that the surrounding coordinate path still lacked material
action utility, so the bridge principle survived but the V98 realization did not
become a healthy endpoint.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；bridge 原则保留，V97/V98 realization 未闭合 action utility。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 544-560, 574-575 and 582-594。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

CI-0066, CI-0068, two meanings of flow and address/value ownership.

### CI-0074 — Reachable gradients can still be unusably weak or semantically indirect

【时间 / 版本】

V97 top-to-bottom gradient audit, 2026-07-22.

【当时在解决什么问题】

Initial action-only backward tests gave zero gradients to most top blocks; after
that was explained, the policy-group path through trajectory-as-evidence still
had gradients of only about `1e-8`.

【当时的 insight / hypothesis】

“A gradient exists” is not functional closure. Initialization gates can delay
upstream learning, and an indirect action-to-evidence-to-action loop can be both
weak and semantically circular even after it becomes differentiable.

【具体做法】

- Separated structural reachability tests from first-training-step behavior.
- Found two final MMDiT residual gates initialized exactly to zero.
- Reused an existing small non-zero residual-init semantic rather than injecting
  artificial gradients.
- Measured the policy group's indirect trajectory-evidence gradient.
- Routed the policy delta directly into the decoder action stream and kept the
  clean proposal in the evidence bank.

【观察到的结果】

All eight blocks received action-loss gradients in the repaired small test. The
policy-group gradient rose from roughly `1e-8` to `1e-4`, while the architecture
retained a single final action generator.

【当时的解释】

The direct route matched ownership: policy deltas belong to action state, not to
an evidence memory that the action decoder must read back from itself.

【后来怎么看（截至冻结终点）】

Partly retained. Later V97/V98 logs confirmed that reachable raw/detail and
address gradients could remain action-irrelevant, while the direct policy-owner
principle itself was not falsified. The version-level benefit of this individual
route was never isolated from the simultaneous address corrections.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；direct-owner principle retained, isolated benefit unverified.

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 551-560 and 575。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Gradient closure, zero residual gates, proposal/evidence ownership and echo
paths.

### CI-0075 — Local-sampling support must be defined in physical cells, not arbitrary pixels

【时间 / 版本】

V97 high-resolution reader review, 2026-07-22.

【当时在解决什么问题】

The raw reader retained `84x84` features but used a fixed `radius=3` pixels
around an `8x8` read lattice whose neighboring centers were about 12 pixels
apart.

【当时的 insight / hypothesis】

Nominal resolution and candidate count do not prove spatial coverage. A support
radius should be expressed relative to the physical/read cell it is supposed to
cover; otherwise high-resolution memory contains systematic blind gaps.

【具体做法】

- Compared the reader's center spacing with its sampled pixel offsets.
- Reinterpreted the `7x7` candidate lattice as normalized offsets within each
  read cell.
- Kept candidate count constant while covering the full cell.

【观察到的结果】

The revised geometry removed unvisited gaps without a multiplicative increase
in candidates.

【当时的解释】

This was a geometry/units bug, not a request for more capacity. The same number
of samples could cover the intended domain if their coordinates were correct.

【后来怎么看（截至冻结终点）】

Retained as a durable implementation rule; later V98 changed the coarse-flow
source but kept continuous local refinement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible message 553。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Coordinate units, N-candidate coverage and false compute savings.

### CI-0076 — Uniform soft correspondence must mean identity, not the image center

【时间 / 版本】

V97 early-log diagnosis and V98 repair, 2026-07-22.

【当时在解决什么问题】

V97 initialized global Flow from random raw features. Near-uniform correlation
was converted to an absolute target-coordinate expectation, producing large
flows toward the center of the image.

【当时的 insight / hypothesis】

When correspondence evidence is uninformative, the neutral element for motion
is identity/zero displacement. A soft expectation over absolute coordinates has
the wrong neutral element and turns uncertainty into systematic motion.

【具体做法】

- Read `LatentSeaRaft`'s global softmax and coordinate expectation.
- Matched the mechanism to `raw_flow 17.8 -> 9.1`, confidence near zero and
  occlusion near one.
- Recentered candidate offsets around each source location.
- Made uniform global and local correlation produce exact zero displacement.
- Applied identity-centered correspondence to both the DINO semantic path and
  the V98 raw-refinement path.

【观察到的结果】

Targeted tests gave zero center drift for uniform correlation and zero residual
at boundaries. The old V97 log was reclassified as an implementation-contract
failure rather than ordinary cold start.

【当时的解释】

More training could shrink the artificial motion, but it would spend early
optimization undoing a deterministic bias and compete with the action loss.

【后来怎么看（截至冻结终点）】

The algebraic repair passed local tests and remains correct as a neutral-address
identity. Later V98 evidence nevertheless rejected the broader address path as
action-useful, so local correctness is not a version-level success claim.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；identity algebra 保留，V98 整体 address utility 未闭环。

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 573-575, 580-582, 589-594 and 604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Zero-flow neutral semantics, uncertainty, dustbin/identity fallbacks and
initialization geometry.

### CI-0077 — Geometric losses require coordinate-invariant units

【时间 / 版本】

V97 early-log loss audit, 2026-07-22.

【当时在解决什么问题】

V96 computed cycle-like quantities on a much smaller DINO grid, while V97 used
an `84x84` raw-feature grid but inherited the same nominal `0.01` loss weight.

【当时的 insight / hypothesis】

Equal coefficients do not mean equal supervision when the represented physical
quantity changes units. Flow, cycle and smoothness should be expressed in a
shared physical/normalized coordinate system before comparing or weighting
them.

【具体做法】

- Decomposed the representation loss into future, cycle, warp and smoothness.
- Compared effective weighted contributions rather than only raw coefficients.
- Found V97 cycle contributed about 20 times V96's early cycle term.
- Related the excess to a rise in global preclip gradient from about `1.08` to
  `1.54` under a global clip of `1.0`.
- Converted V98 geometry losses to a common `8x8` physical grid and added a
  continuous boundary penalty.

【观察到的结果】

The early total-loss gap was dominated by cycle, not JEPA. A large fraction of
V97's gradient budget was therefore spent correcting coordinate-scale artifacts.

【当时的解释】

This was not safely fixed by reducing a weight: the target algebra first needed
the same unit semantics across resolutions.

【后来怎么看（截至冻结终点）】

V98 implemented the normalized-unit repair. Later evidence did not isolate its
behavioral benefit, but no later result reversed the requirement that geometric
losses share one declared coordinate system.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；单位契约保留，孤立收益未被识别。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 573-575, 580-582 and 589-594。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Flow horizon units, global clipping competition and cross-version loss ledgers.

### CI-0078 — Non-zero gradients do not prove that the intended mechanism is identifiable

【时间 / 版本】

V97 extended-log audit, 2026-07-22.

【当时在解决什么问题】

V97's raw pyramid, high-flow, address reader and all three block groups had
non-zero gradients, yet the run did not demonstrate that fine Flow was carrying
useful detail.

【当时的 insight / hypothesis】

A mechanism can be fully differentiable and still be optional. If identity
fallback, low-confidence suppression, smoothness and another coarse path solve
the same losses, the intended fine route may collapse while every parameter
continues to receive some gradient.

【具体做法】

- Compared raw-flow magnitude, confidence, occlusion, entropy and router
  gradients over time.
- Separated coarse/high-flow gradients from detail-router gradients.
- Checked the dataset's strong identity prior as supporting, not definitive,
  evidence.
- Distinguished internal representation-group totals from their signed
  components.

【观察到的结果】

- Early V97 had high-entropy, low-confidence addresses.
- In the extended log, raw flow shrank toward `0.07-0.13`, confidence remained
  about `0.014-0.017`, and detail-router gradient fell to about `2.2e-7`.
- Coarse/high-flow gradients remained roughly `1e-1`, showing compensation by a
  different part of the raw path.
- Negative uncertainty NLL partially cancelled future loss, making the aggregate
  representation group look deceptively small.

【当时的解释】

The implementation had a legal shortcut: reduce Flow's decision relevance and
use identity/coarse evidence. Structural liveness was therefore necessary but
not sufficient evidence of functional use.

【后来怎么看（截至冻结终点）】

V98 removed the independent raw coarse path and stopped attenuating detail
values. Later evidence still failed to establish action-relevant fine-detail
identifiability, so the V97 failure is confirmed while the repair remains only
partial.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；V97 failure 已确认，V98 未建立 fine-detail action closure。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；推断/未决
- 对话：visible messages 575, 582, 589 and 595-604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Optional branches, shortcut learning, aggregate-loss cancellation and causal
interventions.

### CI-0079 — Limit compute through support and lifetime, not information-killing gates

【时间 / 版本】

V97-to-V98 memory and mechanism decision, 2026-07-22.

【当时在解决什么问题】

The raw branch raised memory substantially. The user explicitly rejected
solving that cost by discarding fine detail or using hard gates.

【当时的 insight / hypothesis】

Mechanism constraints should bound the size and lifetime of computation while
preserving a continuous information and gradient path. Confidence-triggered
bypass, hard top-k and detach make “do not use raw detail” the cheapest learned
solution.

【具体做法】

- Removed raw `21x21` global all-pairs matching.
- Kept a fixed number of local deformable samples and output evidence tokens.
- Bounded raw residuals around a DINO coarse address.
- Kept identity/DINO/raw contributions continuously weighted.
- Released the high-resolution container after block three without detaching
  the earlier graph.
- Used activation checkpointing/recomputation and proposed staged memory
  diagnostics instead of intermittent branch execution.

【观察到的结果】

V98 retained `84x84` raw detail while deleting the largest unnecessary global
raw graph. No later evidence in this task isolated the exact memory saving of
that change from the rest of V98.

【当时的解释】

The scarce resource was controlled by locality, fixed query count and activation
lifecycle rather than by erasing optional observations.

【后来怎么看（截至冻结终点）】

Retained as a strong design principle. Later V98 runtime established that the
model could run at production shape, but did not provide a clean before/after
memory attribution for this one lifetime repair.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；mechanism retained, isolated memory saving尚未验证。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 576-585 and 589-594。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Late bottlenecks, soft versus hard routing, memory truth and anti-shortcut
boundaries.

### CI-0080 — Confidence may control address precision, not erase the value it found

【时间 / 版本】

V97 reader audit and V98 repair, 2026-07-22.

【当时在解决什么问题】

The V97 raw reader multiplied high-resolution values by a route/detail factor
of roughly `0.2-0.4`, so uncertain routing also attenuated the content itself.

【当时的 insight / hypothesis】

Address uncertainty and value amplitude are different semantics. Reliability
can widen a posterior, interpolate toward identity or reduce address precision;
it should not make an observed edge or contact feature physically disappear
after it has been read.

【具体做法】

- Traced the route factor from address computation to value multiplication.
- Tested low- and high-route settings with the same raw value.
- Removed amplitude scaling of raw detail.
- Kept continuous controls on address precision and Flow prior only.

【观察到的结果】

Targeted tests showed raw values no longer shrank when routing confidence was
low, while the address mechanism remained continuous.

【当时的解释】

The old multiplication paid for stability with information loss and amplified
the model's incentive to ignore Flow. Separating precision from value preserved
the observation while still expressing uncertainty.

【后来怎么看（截至冻结终点）】

Implemented in V98. Later matched interventions showed that the surrounding
coordinate path remained action-irrelevant; the narrower rule—that confidence
may adjust precision but must not erase retrieved value—survived.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；confidence/value 分权保留，V98 realization 未证明 action utility。

【证据与锚点】

- 类型：源码事实；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 589-594 and 604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Reliability versus loss support, selector/value ownership and repeated
information-amplitude regressions.

### CI-0081 — A failed experiment identity must remain immutable

【时间 / 版本】

V97-to-V98 experiment governance, 2026-07-22.

【当时在解决什么问题】

After repairing V97's contract, reusing the V97 name would have mixed logs and
checkpoints from two behaviorally different graphs.

【当时的 insight / hypothesis】

Version identity is evidence provenance. A materially changed graph needs a new
identity even when it is described as a bug fix; otherwise later comparisons can
silently pair incompatible implementations.

【具体做法】

- Preserved the flawed V97 result as V97.
- Named the corrected graph V98.
- Made the old V97 entry announce/forward rather than overwrite old outputs.
- Separated output directories and log labels.
- Decoupled DataLoader, flow-time and action-noise RNG from model-parameter
  initialization so parameter-count changes would not alter paired samples.

【观察到的结果】

V98 could be compared as a fresh experiment, while V97 remained interpretable
as evidence of the raw-global-matching failure.

【当时的解释】

Shared seed alone had not produced a paired experiment because additional model
parameters consumed the global RNG stream before data/noise sampling.

【后来怎么看（截至冻结终点）】

Retained; actual V98/V96 paired evidence begins after this segment.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 575, 592-594 and 604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

CI-0001, run identity, fresh-run contracts and reproducible counterfactuals.

### CI-0082 — Activation topology, not parameter count, dominated raw-branch memory

【时间 / 版本】

V96/V97 memory audit, 2026-07-22.

【当时在解决什么问题】

The user asked why V96 used only about `9500 MB` and whether low memory implied
that parts of the model were not running.

【当时的 insight / hypothesis】

Memory must be attributed to resident inputs, activation resolution, pair count,
candidate sampling graphs, batch size and counterfactual repetitions. Parameter
count alone is often the wrong explanation.

【具体做法】

- Verified V96 still executed eight top blocks and the three-block decoder.
- Compared cached DINO tensors with V97's `84/42/21` raw pyramid.
- Counted observed RGB frames and temporal camera pairs at batch four.
- Traced FP32 local `grid_sample` graphs and counterfactual construction.
- Compared about `150.89M` versus `152.82M` parameters.

【观察到的结果】

V96's roughly `9.5 GB` was consistent with batch four, cached DINO, low-grid
Flow and no zero-weight duplicate forward graphs. The roughly `1.93M` parameter
difference could not explain V97's high-resolution activation cost.

【当时的解释】

The low V96 number was a useful no-raw baseline rather than evidence of a bypass.
Unexpected growth in comparable future runs should first be localized to raw
pyramid, local correlation and retained sampling graphs.

【后来怎么看（截至冻结终点）】

Retained. Later production-shape work supported activation-lifetime accounting,
but the archive never isolated the memory saving of this one V98 edit from batch,
checkpointing and adjacent raw-path changes.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录；历史检索
- 对话：visible messages 576-579 and 583-594。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`grid_sample`。
- 直接量值/表达式（卡内原文）：`84/42/21`。

【关联】

Computational truth, batch scaling, zero-weight counterfactuals and runtime
instrumentation.

### CI-0083 — A sudden finite-to-nonfinite transition can be a zero-norm derivative singularity

【时间 / 版本】

V96 extended-log numerical failure and V98 shared repair, 2026-07-22.

【当时在解决什么问题】

V96 remained finite through batch 2500 with global preclip norm around `1.14`,
then failed on the next unlogged backward pass with a non-finite gradient.

【当时的 insight / hypothesis】

An abrupt failure after an ordinary finite batch is not necessarily gradual
explosion. Direct `sqrt(sum(x^2))` on trainable vectors has a singular derivative
at exact zero; a model learning near-zero Flow can therefore have finite forward
values and a non-finite backward on one sample.

【具体做法】

- Distinguished the last logged finite batch from the failing unlogged batch.
- Searched Flow magnitude, cycle, confidence and routing paths for direct norms.
- Replaced gradient-path zero-point norms with stable formulations that preserve
  exact zero value and finite/zero derivative semantics.
- Added first-nonfinite-parameter reporting with NaN/Inf counts and finite-part
  maxima.

【观察到的结果】

Targeted zero-flow backward tests became finite. The source mechanism matched
the failure timing closely, though the old log lacked a first-bad-parameter
record and therefore could not prove the exact historical tensor.

【当时的解释】

The conclusion was deliberately stated as a high-confidence structural cause,
not perfect post-hoc proof. The new sentinel was necessary to make a recurrence
directly attributable.

【后来怎么看（截至冻结终点）】

Implemented across the shared V96/V98 Flow path; production recurrence remained
to be tested.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；numerical repair verified locally, historical attribution probabilistic.

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 595-604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。

【关联】

Non-finite sentinels, zero semantics and late-training numerical failures.

### CI-0084 — Exact terminal identity should reuse computation, not merely repeat its formula

【时间 / 版本】

V96/V97 terminal-semantics audit, 2026-07-22.

【当时在解决什么问题】

Both logs showed `terminal_identity_error` around `2.4e-3` even though the
terminal candidate was intended to be a no-op copy of the prefix velocity.

【当时的 insight / hypothesis】

Mathematical formula equality does not guarantee bitwise semantic identity in
BF16 when the same prefix is recomputed with a different batch shape through
LayerNorm/Linear kernels. If a contract requires exact identity, the value must
be reused, not independently reconstructed.

【具体做法】

- Traced the terminal path and ruled out an actual terminal operation.
- Compared prefix recomputation under different batch shapes.
- Reused the already-computed prefix velocity for the terminal candidate.
- Added elementwise identity tests in BF16.

【观察到的结果】

The local BF16 test became exactly equal in value and gradient semantics.

【当时的解释】

The old discrepancy was numerical duplication, not conceptual terminal motion.
The repair made the zero/no-op contract explicit and cheaper.

【后来怎么看（截至冻结终点）】

Implemented and locally verified. No later archive entry isolated this diagnostic
as a production-performance factor; the durable requirement is reuse of the
canonical computation for an exact identity check.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；局部数值闭环已验证，不外推独立生产收益。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；历史检索
- 对话：visible messages 595-604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`terminal_identity_error`。

【关联】

Terminal/no-op semantics, BF16 reproducibility and reuse versus recomputation.

### CI-0085 — An audit must treat fatal termination and signed loss composition as first-class evidence

【时间 / 版本】

V96/V97 audit-tool correction, 2026-07-22.

【当时在解决什么问题】

The automated audit read V96's traceback but summarized the run mainly through
its last finite metric line, while V97's aggregate representation loss appeared
small because a negative uncertainty term cancelled future error.

【当时的 insight / hypothesis】

The last metric row is not the run outcome, and a group total is not a health
metric when signed components can cancel. Fatal tracebacks, OOM/non-finite exits
and component-wise contributions must outrank ordinary trend summaries.

【具体做法】

- Added explicit traceback, OOM, non-finite loss and non-finite backward
  detection to the audit tool.
- Re-ran the V96 log and classified it as
  `critical numerics/non-finite-backward`.
- Compared V97 future and uncertainty-NLL components rather than relying on the
  small representation-group total.
- Added the historical failure as an audit regression fixture.

【观察到的结果】

The corrected audit no longer called the V96 run merely incomplete and no longer
treated V97's small aggregate representation loss as proof of a healthy world
model.

【当时的解释】

Audit semantics are part of experimental correctness. A parser that suppresses
the terminal event or permits cancellation to hide a failing component can
reverse the engineering decision.

【后来怎么看（截至冻结终点）】

Retained as a durable audit requirement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 595-604。
- 段级工件：见 Segment 008 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`critical numerics/non-finite-backward`。

【关联】

CI-0004, loss ledgers, failure sentinels and incomplete-run interpretation.

## Segment 008 synthesis

This segment began by correcting two naming illusions:

```text
eight "role" blocks  != eight causally specialized blocks
native DINO detail   != raw-image detail
```

V97 then made both boundaries real: a `3+3+2` grounding/world/policy stack and a
raw pyramid feeding a continuous reader. Static and small-backward tests showed
that the new graph was connected, but logs immediately falsified a more specific
claim: the raw path did not implement the intended DINO-seeded physical
refinement. Random raw features performed global soft matching, uniform
correlation pulled coordinates toward the image center, `84x84` geometry reused
loss units from much smaller grids, and uncertainty attenuated the detail value
itself. The fine router could therefore become almost irrelevant while coarse
raw gradients remained non-zero.

The V98 correction preserved the information-bearing decisions and replaced the
faulty mechanism:

```text
DINO identity-centered coarse address
    -> bounded 42/84 raw residual
    -> continuous high-resolution read
    -> fixed evidence boundary after grounding
```

It also separated compute control from information deletion, isolated experiment
RNG, preserved V97 as an immutable failed identity, stabilized exact-zero norms,
made terminal no-op reuse exact and taught the audit tool to recognize fatal
backward failures. At message 604 these changes had only source/local-test
evidence. Later production V98 evidence showed better terminal/detail-router
activity and strong final action numbers, but causal interventions also showed
that its sample-specific Flow coordinates changed representation without
materially changing action. The mechanism was therefore only partially closed.

## Segment 009 — Production-shape smoke, initialization provenance and complete log scope

### Segment boundary

```text
Date:                 2026-07-22
Visible messages:     605-643
Starts with:          decision to require a smoke before another long run
Ends with:            early V98 evidence compared against an invalid warm-start V96 baseline
Primary historical artifacts:
  scripts/current_v98_dino_seeded_raw_flow_332_jepa_smoke.sh
  scripts/current_v98_dino_seeded_raw_flow_332_jepa.sh
  scripts/current_v96_late_bottleneck_jepa.sh
  attached extended V96/V97 logs
  attached early batch-eight V98/V96 logs
```

### CI-0086 — A smoke must preserve production shape and lifecycle

【时间 / 版本】

V98 release preflight, 2026-07-22.

【当时在解决什么问题】

After V96's non-finite failure and V97's structurally wrong raw path, the user
required a smoke before another long experiment.

【当时的 insight / hypothesis】

A tiny batch or simplified forward is not evidence that the production run is
safe. A useful smoke shortens dataset coverage while preserving batch shape,
BF16, activation checkpointing, raw-reader geometry, backward, clipping,
optimizer, endpoint semantics and checkpoint isolation.

【具体做法】

- Added a dedicated V98 smoke wrapper rather than relying on a hand-written
  argument tail.
- Kept the then-formal batch four, full `3+3+2` graph and BF16 path.
- Reduced only train/validation batch counts.
- Logged loss, gradients and staged memory every batch.
- Prohibited resume and output-directory reuse.
- Explicitly warned that a batch-four smoke could not validate a later
  batch-eight decision.

【观察到的结果】

Shell/preflight checks closed the wrapper chain, but actual CUDA behavior still
required the server. The subsequent batch-size change correctly invalidated the
earlier memory assurance.

【当时的解释】

The smoke was an experiment-contract check, not a toy functional test. Keeping
production tensor geometry was necessary to expose raw-branch memory and
backward failures.

【后来怎么看（截至冻结终点）】

Retained as a durable release rule. Later schema work repeatedly returned to
production-batch smoke and fresh-run gates.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 605-612 and 620-621。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。

【关联】

CI-0057, deployment preflight, memory truth and fresh-run isolation.

### CI-0087 — Shell inheritance is part of the realized run context

【时间 / 版本】

V98 smoke launch, 2026-07-22.

【当时在解决什么问题】

The V98 smoke launched successfully even though the invocation did not visibly
specify the required T5 `.pt` path.

【当时的 insight / hypothesis】

An omitted command-line value may still be supplied by the parent process
environment. Reproducibility therefore requires the resolved run context, not
just the visible command copied into chat.

【具体做法】

- Traced `T5_CONDITION_PATH` through `nohup`, wrapper scripts and Python.
- Verified the startup path had a hard file-existence check.
- Provided `echo`, `readlink` and log-banner checks for the resolved artifact.
- Proposed an `env -u` negative test to prove the requirement was real.

【观察到的结果】

The command inherited a previously exported T5 path; language conditioning had
not silently disabled itself.

【当时的解释】

The discrepancy was shell state, not model fallback. It demonstrated why a run
must serialize resolved paths and fingerprints rather than rely on launch text.

【后来怎么看（截至冻结终点）】

Retained. Later mainline manifests made resolved artifact identity a first-class
startup diagnostic.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；方案/设计记录
- 对话：visible messages 613-615。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`T5_CONDITION_PATH`。

【关联】

Run-context closure, T5 fingerprinting and hidden wrapper/environment state.

### CI-0088 — A checkpoint path in arguments is not proof that weights were loaded

【时间 / 版本】

V96/V97/V98 initialization audit, 2026-07-22.

【当时在解决什么问题】

Old `best_contract.pt` paths remained visible in inherited script arguments,
creating conflicting claims about whether several runs were fresh.

【当时的 insight / hypothesis】

Checkpoint provenance must distinguish a carried path from the final enable
switch, resolved runtime checkpoint, load report and resulting parameter match.
A static argument summary can falsely label a fresh run as warm-started—or hide
a real partial load.

【具体做法】

- Traced `stage1_initialization_enabled` through final wrapper precedence.
- Checked serialized `stage1_checkpoint` rather than the inherited raw path.
- Read skipped-shape, missing-key and initialization reports.
- Corrected the old V96/V97 pair to fresh after finding
  `stage1_initialization_enabled=false` and runtime checkpoint `null`.
- Separately identified the newer V96 as a true partial warm start.

【观察到的结果】

The old comparison logs were fresh despite carrying a dormant path. The newer
V96 loaded a legacy checkpoint, skipped 31 shape mismatches and left 553
parameters missing.

【当时的解释】

The same visible filename had two different realized meanings in two runs.
Only the final load event established experimental identity.

【后来怎么看（截至冻结终点）】

Retained as a central provenance rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 622-624, 631-634 and 638-643。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`best_contract.pt`、`stage1_initialization_enabled`、`stage1_checkpoint`。

【关联】

CI-0001, CI-0081, exact resume rejection and run fingerprints.

### CI-0089 — Partial warm start is a distinct intervention, not a neutral baseline

【时间 / 版本】

V96/V98 paired-run preparation, 2026-07-22.

【当时在解决什么问题】

The inherited V40 checkpoint partially matched V96. The user asked whether its
contents would actually matter.

【当时的 insight / hypothesis】

`strict=False` loading can change optimization semantics even when optimizer,
scheduler and epoch restart at zero. Old trunk and adapters may preserve a prior
solution while new Flow/JEPA/language modules begin random, making the new
modules adapt around an inherited shortcut.

【具体做法】

- Mapped which same-name/same-shape modules could load.
- Distinguished deleted/mismatched decoder weights from inherited trunk,
  evidence and contract components.
- Noted that dirty adapters were not reset in that V96 launch.
- Reclassified the run as `V96-warmstart` rather than a V96 architectural
  baseline.
- Required a separate scratch V96 for fair comparison.

【观察到的结果】

The warm-start run could plausibly converge faster or look better early, while
leaving many new modules random. Its results could not attribute gains to V96's
new architecture.

【当时的解释】

Warm start was potentially useful as its own ablation, but not interchangeable
with fresh initialization.

【后来怎么看（截至冻结终点）】

Retained. The early V98 log subsequently outperformed this warm-start V96 on
several action-training metrics, but that did not repair baseline validity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 624-627 and 638-643。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。

【关联】

Optimizer state versus model state, migration reports and architecture
attribution.

### CI-0090 — Failure diagnosis is not a complete experiment audit

【时间 / 版本】

V96/V97 extended-log re-audit, 2026-07-22.

【当时在解决什么问题】

The user pointed out that the prior review had concentrated on V96's numerical
failure and V97's raw-flow defect instead of examining all available training,
JEPA, language, execution and gradient evidence.

【当时的 insight / hypothesis】

Finding a decisive error answers whether a run is releasable; it does not answer
what useful structure the run demonstrated, which subsystems remained healthy,
or what later versions should preserve.

【具体做法】

- Reopened the original logs rather than extending the previous error summary.
- Separated training-only claims from unavailable validation claims.
- Used common-step windows and medians instead of last-row comparisons.
- Decomposed action horizons, JEPA geometry, raw routing, language/history,
  execution/value and gradient ownership.
- Preserved both positive and negative evidence from the rejected V97 run.

【观察到的结果】

V97 had no overall action-loss advantage and its fine raw router was nearly
inactive, yet it showed better tail/first balance and lower adjacent-horizon
cosine than V96. V96 was cleaner geometrically and stronger near-term but more
temporally homogeneous.

【当时的解释】

The first audit was too narrow. The user's correction changed the review scope
from “why this run is invalid” to “what the run taught us despite invalidity.”

【后来怎么看（截至冻结终点）】

Retained as a process correction. Later cross-version audits increasingly
required full metric families and preservation of useful failed-version
evidence.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；prior review scope explicitly corrected.

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 628-637。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。

【关联】

Historical versions as analytical lenses, matched-window comparison and
failure-versus-learning distinction.

### CI-0091 — Useful temporal organization can coexist with a failed detail mechanism

【时间 / 版本】

V96/V97 extended comparison, 2026-07-22.

【当时在解决什么问题】

It was unclear whether V97's defective raw-flow path invalidated the entire
`3+3+2` and raw-evidence direction.

【当时的 insight / hypothesis】

A version is not one indivisible mechanism. A broken producer can coexist with
a useful downstream organization change, so repair decisions should follow
field-level evidence rather than accept/reject the whole version.

【具体做法】

- Compared V96/V97 at batch 2300-2500 medians.
- Separated first-eight and tail errors from aggregate action loss.
- Compared future/warp/cycle errors with adjacent-horizon cosine.
- Audited raw flow, confidence, fallback mass, detail gate and detail-router
  gradients independently.

【观察到的结果】

- Aggregate action loss was essentially tied.
- V96 first-eight was stronger but tail weaker; V97 had tail/first near one.
- Adjacent-horizon cosine was about `0.978` for V96 and `0.934` for V97.
- V97 raw confidence stayed near `0.01-0.015`, fallback exceeded flow mass and
  detail-router gradient was roughly `2.3e-7`.

【当时的解释】

V97 offered evidence for improved temporal differentiation while failing to
make raw Flow a reliable detail address. The former was worth preserving through
the latter's repair.

【后来怎么看（截至冻结终点）】

V98 was explicitly judged by whether it retained horizon/tail balance while
repairing detail routing and terminal identity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V97 mechanism split into retained and rejected parts.

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 633-637。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。

【关联】

CI-0078, modular causal attribution and later historical replay methodology.

### CI-0092 — Early V98 evidence showed repair, not final ownership

【时间 / 版本】

V98 scratch early run, batch eight, 2026-07-22.

【当时在解决什么问题】

The first production-shaped V98 log needed to show whether V97's known defects
were actually repaired before continuing.

【当时的 insight / hypothesis】

Repair evidence should be tied to the failed mechanism's direct diagnostics.
Better aggregate loss is supportive but cannot prove address ownership,
long-horizon use or generalization.

【具体做法】

- Checked terminal identity, raw detail-router gradient, confidence, valid
  coverage, flow/fallback mass and horizon losses.
- Compared common batch 400-580 against the available warm-start V96 only as a
  directional reference.
- Kept gripper, DINO-seed reliability and weak world-block gradients as open
  observations.

【观察到的结果】

- Terminal identity error disappeared.
- Detail-router gradient rose from the old V97's `~2e-7` to `1e-5-1e-4`.
- Confidence and valid coverage rose, and high residual remained non-zero.
- V98 showed balanced first/tail training error and lower horizon cosine.
- Flow address mass remained only about `0.39-0.43`, fallback stayed dominant,
  entropy remained high and no validation epoch existed.

【当时的解释】

The known gradient/identity failures were repaired, but “Flow owns the spatial
address” was still too strong. The path had become effective conditioning, not a
demonstrated decisive coordinate owner.

【后来怎么看（截至冻结终点）】

Retained as provisional early evidence. The next segment reopened whether Flow
would later collapse despite this healthy start.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；direct repairs supported, long-run health unresolved.

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 638-643。
- 段级工件：见 Segment 009 boundary 的 `Primary historical artifacts`。

【关联】

CI-0080, address/fallback semantics, gripper imbalance and world-role debate.

## Segment 009 synthesis

This segment tightened experimental semantics before interpreting another
architecture result:

```text
short smoke != simplified graph
path present != checkpoint loaded
training restarted != model initialized fresh
fatal error found != complete experiment audited
```

The user explicitly corrected the last failure: the first V96/V97 review had
explained why the runs were invalid but had not extracted all of their useful
evidence. The full re-audit recovered a more nuanced result—V97's fine raw route
was functionally weak, yet its temporal organization appeared less homogeneous
and more balanced across action horizons. Early V98 then showed that terminal
identity and detail-router gradients had genuinely improved, while address
ownership, gripper learning, seed reliability, world-role strength and
validation remained open. The simultaneous V96 launch was a partial legacy warm
start, so it could not settle architecture ranking.

## Segment 010 — Runtime semantics, mixed Flow address and the distributed-world question

### Segment boundary

```text
Date:                 2026-07-22
Visible messages:     644-675
Starts with:          clarification of training-batch time versus deployment inference
Ends with:            mature pre-validation V98 audit and decision to defer world-role changes
Primary historical artifacts:
  V98 batch-eight training log through batch 1540
  warm-start V96 log through batch 2560
  clearvla/policy/trunk_primitives.py
  clearvla/policy/trunk.py
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/flow_dino_evidence.py
```

### CI-0093 — Training throughput and deployment latency are different workloads

【时间 / 版本】

V98 runtime interpretation, 2026-07-22.

【当时在解决什么问题】

The user asked whether one logged training batch was approximately one deployed
inference and whether V98 counted as a lightweight base model.

【当时的 insight / hypothesis】

Wall-clock numbers are meaningful only after expanding the lifecycle they time.
Batch-eight training with multiple forwards, losses, backward and optimizer is
not directly divisible into batch-one five-step deployment latency.

【具体做法】

- Enumerated the main and counterfactual training forwards, backward, clipping
  and optimizer work.
- Distinguished one-time visual-context encoding from five serial policy
  sampling evaluations.
- Separated trainable parameter count from online DINO, raw-flow and candidate
  read cost.
- Marked the `0.3-0.8 s/action chunk` estimate as an estimate requiring a
  synchronized deployment benchmark.

【观察到的结果】

V98 was light relative to billion-parameter VLAs but not light relative to
edge-device visual policies. Its roughly 153M trainable parameters omitted the
cost of any online DINO encoder and did not predict five-step latency.

【当时的解释】

“Lightweight” was workload-relative. Action-chunk replanning at low frequency
could be practical while high-rate per-frame control remained unproven.

【后来怎么看（截至冻结终点）】

Retained as a measurement rule; no deployment benchmark had yet replaced the
estimate.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；numeric latency estimate remained unverified.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 644-650。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.3-0.8 s/action chunk`。

【关联】

Static versus dynamic lifecycle, five-step sampling and activation-cost truth.

### CI-0094 — Optimize cost only after proving the mechanism that cost belongs to

【时间 / 版本】

V98 experiment priority decision, 2026-07-22.

【当时在解决什么问题】

V98 had clear latency and raw-reader optimization opportunities, but the user
preferred to make the intended mechanism real before compressing it.

【当时的 insight / hypothesis】

Premature efficiency work can erase the very route whose usefulness has not yet
been identified. First prove action, long-horizon and detail-path function;
then optimize sampling, online vision and compilation against those invariants.

【具体做法】

- Prioritized a full epoch and validation over reducing steps or tokens.
- Kept scratch initialization and matched experiment controls as prerequisites.
- Listed action horizon, gripper event, JEPA, Flow address, world gradient,
  execution and non-finite gates.
- Deferred distillation, fewer sampling steps and reader compression.

【观察到的结果】

The run continued without an efficiency-driven architecture change.

【当时的解释】

An unverified mechanism cannot supply a safe target for optimization; otherwise
the project cannot tell whether a speedup removed redundancy or useful signal.

【后来怎么看（截至冻结终点）】

Retained and echoed in later “information preservation before purity/efficiency”
requirements.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 651-652。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

Information conservation, deployment optimization and closed-loop validation.

### CI-0095 — “Half error” names incomplete semantics, not merely an early metric

【时间 / 版本】

V98 early-risk classification, 2026-07-22.

【当时在解决什么问题】

The user asked which concerns were real implementation errors, which had only
half-realized their intended function and which were ordinary early-training
observations.

【当时的 insight / hypothesis】

A half error is a live, differentiable implementation whose realized ownership
or semantics cover only part of the adopted design. It should not be inferred
from one low gradient, one early loss or an unsigned trend.

【具体做法】

- Separated the definite V96 initialization error and old V97 identity/router
  errors from provisional V98 observations.
- Classified world-role ownership, homogeneous Goal tokens, unaligned
  language/action carriers, mixed Flow address and unobserved detail-gate
  selectivity as partial-closure candidates.
- Kept early gripper lag, shrinking Flow, execution warmup and negative NLL out
  of the error category pending later evidence.

【观察到的结果】

The discussion avoided treating every imperfect magnitude as a defect. It also
made explicit that “gradient exists” and “design role is fulfilled” were
different claims.

【当时的解释】

The classification tied error status to the intended contract and available
evidence rather than dissatisfaction with a number.

【后来怎么看（截至冻结终点）】

Some provisional concerns changed quickly: arm/gripper balance improved, while
near-zero Flow became a stronger structural issue by batch 1540.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留 as a reasoning framework; individual classifications remained revisable.

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 653-656 and 674-675。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

Functional closure, early-log uncertainty and later “trivial/non-trivial”
repair distinctions.

### CI-0096 — V98 had a mixed spatial address, not “no address”

【时间 / 版本】

V98 address semantics clarification, 2026-07-22.

【当时在解决什么问题】

An earlier summary said Flow had not yet obtained a spatial address because its
mass was below fallback and entropy was high. The user asked what the reader had
actually obtained, then.

【当时的 insight / hypothesis】

Address existence, sharpness and causal ownership are separate properties. A
soft mixture of stationary and advected local coordinates is still a spatial
address; what remained unproved was whether correct Flow was selectively useful
on motion regions.

【具体做法】

- Expanded the 98 candidates into 49 identity-local and 49 Flow-advected local
  samples per reader cell.
- Interpreted fallback as an Eulerian stationary coordinate, not a random null.
- Interpreted Flow as a motion-advected coordinate.
- Listed the missing conditional diagnostics: motion/static mass, confidence,
  occlusion fallback, zero/shuffle interventions and attention-centroid motion.

【观察到的结果】

The early reader placed roughly 58% mass on stationary local candidates and 42%
on advected candidates while retaining high-resolution anchors and content
attention. Entropy near `0.88` showed diffuse selection but did not erase spatial
identity.

【当时的解释】

The corrected claim was: V98 had a content-conditioned spatial distribution;
it lacked evidence that Flow owned precise address decisions when it mattered.

【后来怎么看（截至冻结终点）】

By batch 1540, Flow/fallback mass approached half-and-half while the displacement
itself approached zero, proving that lane mass alone could not establish motion
address usefulness.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；wording corrected and causal test left open.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 657-658 and 674-675。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

CI-0092, identity fallback, selector mass and address interventions.

### CI-0097 — Top-policy preparation and the bottom action solver are distinct

【时间 / 版本】

V98 role-path source audit, 2026-07-22.

【当时在解决什么问题】

Calling the final two top blocks “policy blocks that write trajectory” made it
sound as if they were the complete policy, prompting the user's correction that
the bottom `3x2` stack still existed.

【当时的 insight / hypothesis】

Module names must not collapse a continuous producer-consumer chain. The top
blocks prepare an action workspace; the bottom Evidence-MMDiT is the actual
iterative action solver.

【具体做法】

- Reconstructed the full top-eight to Evidence Adapter/Organizer to bottom
  decoder path.
- Clarified that the top-policy residual enters action with a fixed lift before
  the bottom solver.
- Verified the bottom has three independently parameterized blocks.
- Clarified that `x2` means maximum dwell/reuse, not six parameter blocks.
- Distinguished the 18 update diagnostics from block count.

【观察到的结果】

At that time soft dwell was about `1.11` and hard dwell `1.0`, so execution was
closer to three bottom operations than six. The bottom simultaneously read
rollout, raw evidence, state, transition, contracts, trajectory and noisy action.

【当时的解释】

The user's objection corrected an imprecise architectural description. It also
made the world-route problem clearer: the bottom solver had several legitimate
alternative evidence sources.

【后来怎么看（截至冻结终点）】

Retained. Later replay work repeatedly distinguished preparation, routing and
physical action ownership.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；assistant wording explicitly corrected.

【证据与锚点】

- 类型：源码事实；过程/决策记录
- 对话：visible messages 659-667。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

Bottom ABI, dwell semantics, action-workspace ownership and bypass analysis.

### CI-0098 — Weak dedicated world blocks do not prove weak world information

【时间 / 版本】

V98 world-role debate, 2026-07-22.

【当时在解决什么问题】

World-block gradient was roughly an order of magnitude below top-policy
gradient. The first explanation treated this as insufficient world ownership;
the user proposed that world information might instead be distributed through
the entire top and bottom path.

【当时的 insight / hypothesis】

World is an information type, not necessarily a fixed layer range. Low
incremental gradient in blocks 4-6 is compatible with useful prior construction,
distributed representation or an actual bypass; logs alone cannot select among
those explanations.

【具体做法】

- Traced role write permissions and found world blocks could only modify
  rollout in late-bottleneck mode.
- Traced grounding's direct raw-detail read and top-policy's direct action
  workspace.
- Traced the bottom solver's many evidence inputs.
- Formulated three competing hypotheses: benign redundancy, distributed world
  representation and harmful bypass.
- Proposed same-checkpoint identity/shuffle interventions at block-3 and block-6
  boundaries before changing the architecture.

【观察到的结果】

Future JEPA improved strongly despite persistently weak world-block gradients.
This proved the full future path learned something, but not which blocks owned
it or whether action depended on it.

【当时的解释】

Dedicated layers might be unnecessary while typed, supervised and intervenable
world state remained necessary. Role labels were not evidence of functional
ownership.

【后来怎么看（截至冻结终点）】

The three explanations were never isolated by the required same-checkpoint
ablation in the frozen archive. The claim therefore remains an evidence warning,
not a conclusion that weak dedicated blocks were healthy or unhealthy.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：未闭环
- 说明：尚未验证；冻结档案没有完成所需的同-checkpoint 因果消融。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；过程/决策记录；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 659-675。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

CI-0070, distributed representation, causal ablation and later S/W ownership
debates.

### CI-0099 — Improving static future error can coexist with losing motion

【时间 / 版本】

V98 run through batch 1540, 2026-07-22.

【当时在解决什么问题】

As V98 matured, action and future-JEPA losses improved, yet raw Flow magnitude,
seed reliability and correlation quality all moved toward an identity solution.

【当时的 insight / hypothesis】

On mostly static data, future semantic prediction, cycle validity and address
lane usage can all look healthy while motion-specific information disappears.
Dynamic health needs independent motion-region evidence, not global future loss
or Flow-lane mass.

【具体做法】

- Followed raw displacement, seed reliability, correlation entropy/margin,
  validity, occlusion, cycle and boundary together.
- Compared horizon error with change-direction and adjacent-horizon cosine.
- Distinguished the Flow candidate lane from the displacement carried by that
  lane.
- Kept the weak-world debate secondary to the emerging static-copy risk.

【观察到的结果】

From batch 20 to 1540, raw grid Flow fell from about `0.698` to `0.013`, seed
reliability from `0.363` to `0.070`, while valid coverage rose and cycle/boundary
fell. Flow mass approached `0.487`, but it mostly selected an almost-identity
coordinate. Future loss improved to about `0.193`, while adjacent-horizon cosine
remained around `0.958` and change-direction about `0.56`.

【当时的解释】

The model could be predicting shared static semantics increasingly well while
discarding motion. This was a stronger immediate concern than whether three
blocks deserved the “world” name.

【后来怎么看（截至冻结终点）】

This observation directly triggered the V99 observable-motion closure in the
next segment.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V98 near-zero-motion risk strengthened by multi-metric evidence.

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 671-675。
- 段级工件：见 Segment 010 boundary 的 `Primary historical artifacts`。

【关联】

CI-0063, static-copy shortcuts, moving-region diagnostics and V99.

## Segment 010 synthesis

The segment repeatedly corrected overloaded shorthand:

```text
training batch     != deployment inference
fallback address   != absence of spatial address
top-policy blocks  != final action solver
weak world blocks  != proof of weak world information
```

The user supplied two important corrections. First, asking what the mixed
Flow/fallback reader actually represented replaced the overstatement “no
address” with a precise stationary-plus-advected distribution whose causal Flow
utility remained unproved. Second, pointing out the bottom `3x2` solver corrected
an architectural description that had collapsed preparation and action solving
into one “policy” label. The later V98 log then shifted priority again: world
layer ownership remained a legitimate ablation question, but the more concrete
risk was that global future metrics improved while displacement and semantic
seed quality collapsed toward static identity.

## Segment 011 — V99 observable motion and causal proof of V98 address irrelevance

### Segment boundary

```text
Date:                 2026-07-22
Visible messages:     676-740
Starts with:          source-level diagnosis of V98's near-zero visual Flow
Ends with:            V99 epoch-one evidence and unresolved downstream ownership
Primary historical artifacts:
  clearvla/policy/flow_dino_evidence.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  scripts/current_v99_observable_raw_flow_332_jepa.sh
  V99 early and epoch-one logs
  V98 zero/shuffle intervention specification and result
  V96/V98/V99 comparison logs
```

### CI-0100 — V98 made zero Flow a jointly optimized solution, not merely a data prior

【时间 / 版本】

V98 mature-flow audit, 2026-07-22.

【当时在解决什么问题】

After the earlier dataset probe had correctly rejected a magnitude-only collapse
claim, mature V98 showed a much stronger combination of near-zero displacement,
degrading correspondence and apparently improving geometric diagnostics.

【当时的 insight / hypothesis】

Sparse true motion does not excuse an objective in which cycle, smoothness,
boundary, trainable descriptors, self-defined motion masks, reliability scaling
and duplicate fallback all cooperate to make zero Flow optimal.

【具体做法】

- Rebuilt the raw-flow loss and reader path from fixed inputs to action.
- Verified history frames were distinct, ruling out an obvious indexing reuse.
- Found cycle, smoothness and boundary all minimized by zero Flow.
- Found warp supervision used the same learnable raw pyramid, allowing descriptor
  and Flow to co-adapt toward similarity.
- Found motion emphasis depended on predicted Flow/reliability and vanished as
  Flow weakened.
- Found low reliability multiplied displacement toward zero.
- Found identity-local and Flow-local candidate banks became duplicates when
  Flow approached zero.

【观察到的结果】

V98's grid Flow fell from about `0.698` to `0.013` while correlation entropy
rose, margin and seed reliability fell, and cycle, boundary, validity and
occlusion appeared to improve. Flow gradients stayed live, showing optimization
rather than a detached/dead branch.

【当时的解释】

The data supplied a static-background bias, but the architecture amplified it
into a legal shortcut and removed downstream pressure for correct motion.

【后来怎么看（截至冻结终点）】

The V98 zero/shuffle intervention later in this segment supplied direct evidence
that its learned coordinates had essentially no action value.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V98 shortcut established by source, trends and intervention.

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；数据探针
- 对话：visible messages 676-681 and 710-736。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

CI-0063, CI-0099, self-referential supervision and optional address routes.

### CI-0101 — Anti-collapse supervision needs an observation-fixed comparator

【时间 / 版本】

V99 design and implementation, 2026-07-22.

【当时在解决什么问题】

The user required the shortcut and the constraint geometry to be repaired
together, without imposing artificial non-zero motion.

【当时的 insight / hypothesis】

The correct target is not a minimum Flow magnitude. A predicted warp should beat
the identity warp where fixed observation evidence shows change, while exact
zero remains valid on static regions.

【具体做法】

- Replaced co-adaptive warp evidence with fixed RGB/census descriptors.
- Defined observable-motion support independently of predicted Flow.
- Added an identity-advantage term only on observable-change regions.
- Kept the new constraint exactly zero on static pairs.
- Changed low reliability from displacement shrinkage to wider continuous
  search.
- Replaced duplicate identity-local fallback with a semantically distinct
  pooled-content fallback.
- Added moving/static/global gain and motion-region correlation diagnostics.

【观察到的结果】

Synthetic static pairs produced exact zero constraint and zero Flow-update
gradient; translated pairs produced positive identity advantage and non-zero
update gradient. BF16 and V98 compatibility paths remained finite.

【当时的解释】

Fixed observation evidence prevented the producer from altering both prediction
and judge. Static zero semantics were preserved rather than penalized.

【后来怎么看（截至冻结终点）】

V99 logs showed moving-region warp gain becoming positive, supporting the
principle, though downstream action use remained open.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；implemented and received early empirical support.

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 682-693。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

Observable support, Teacher isolation, loss masks and anti-shortcut design.

### CI-0102 — Review must remove dependencies that happen to be harmless in the current run

【时间 / 版本】

V99 implementation review, 2026-07-22.

【当时在解决什么问题】

Repeated review found several issues that did not change the active V99 result
but would invalidate a future configuration or backward pass.

【当时的 insight / hypothesis】

A dormant incorrect dependency is still a contract defect. Current objective
coincidence, successful autograd or repeated computation does not make it safe.

【具体做法】

- Added an absolute noise/change floor so relative normalization would not
  promote sensor noise into motion.
- Removed in-place reuse of the same address-gate tensor.
- Removed the false requirement that history-only identity advantage needed
  future observations.
- Avoided rebuilding fixed mid-resolution descriptors.
- Verified the fixed observation graph could not co-adapt through gradients.

【观察到的结果】

The false future dependency would have been hidden because JEPA already loaded
future observations in this run. The review removed it before a future-loss-off
configuration could expose it.

【当时的解释】

The aim was semantic closure under neighboring configurations, not merely a
passing active configuration.

【后来怎么看（截至冻结终点）】

Retained as an example of why multiple producer-to-consumer reviews are needed.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 686-692。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

Mandatory subsystem familiarity, dormant paths and configuration ABI.

### CI-0103 — Zero and shuffle answer different causal questions

【时间 / 版本】

V99 diagnostic design and V98 retrospective intervention, 2026-07-22.

【当时在解决什么问题】

A forced-zero comparison could show sensitivity to displacement but could not
show that sample-specific coordinates were correct rather than merely non-zero.

【当时的 insight / hypothesis】

Zero tests whether motion information matters; shuffle tests whether the
particular sample-aligned motion matters. Both must reuse weights, input and
sampling noise, and neither should alter the graph's other semantics.

【具体做法】

- Added zero and camera-preserving shuffled Flow as validation-only diagnostics.
- Required paired identical action-sampling noise.
- Designed a V98-specific audit switch because enabling V99's guard would have
  changed descriptors, fallback and refiner semantics.
- Required patched baseline to be bit/numerically identical to ordinary
  baseline.
- Fixed the independent evaluator's missing raw-image loading requirement in
  the intervention specification.

【观察到的结果】

The V98 probe successfully perturbed raw-reader values while leaving all other
run semantics stable.

【当时的解释】

An intervention belongs to the graph being audited. Importing a later fix into
an older checkpoint would confound the causal question.

【后来怎么看（截至冻结终点）】

Retained and later generalized into matched interventions for S/W/P and typed
lanes.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 689-693 and 715-734。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

Counterfactual closure, checkpoint identity and matched noise.

### CI-0104 — V99 initially repaired identifiability without yet proving correspondence sharpness

【时间 / 版本】

V99 batch 20-480 smoke/early run, 2026-07-22.

【当时在解决什么问题】

The first V99 log had to show whether the observable-motion constraint prevented
an immediate repeat of V98's collapse.

【当时的 insight / hypothesis】

Positive moving-region gain is stronger evidence than a large Flow magnitude,
but it does not prove that the correlation volume has learned a sharp match or
that static regions are already safe.

【具体做法】

- Compared moving, static and global warp gain against identity.
- Checked observable-motion fraction for mask collapse.
- Followed moving-region entropy/margin separately from global statistics.
- Separated wall-clock slowdown from concurrent GPU contention.

【观察到的结果】

By batch 480, moving gain changed from about `-0.0467` to `+0.0235` and motion
support stayed near `0.37`. Grid Flow shrank but not to zero. Static and global
gain remained slightly negative, and moving entropy/margin did not improve.

【当时的解释】

The repair had made useful moving-region Flow learnable without forcing motion,
but the source of improvement could still be seed/refinement rather than a
well-resolved correlation volume.

【后来怎么看（截至冻结终点）】

Epoch-one and partial epoch-two evidence later showed moving/static gains
improving further, but address entropy remained very high.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；identifiability improved, full address health unproved.

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 700-709 and 737-740。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

Motion-conditioned logging, resource contention and correlation quality.

### CI-0105 — V98 coordinates changed representation but not task performance

【时间 / 版本】

V98 epoch-two zero/shuffle intervention, step 5692, 2026-07-22.

【当时在解决什么问题】

Logs suggested that V98's Flow address had collapsed, but the project needed a
same-checkpoint causal test of whether its coordinates affected action quality.

【当时的 insight / hypothesis】

Internal tensor sensitivity is not downstream utility. A route can substantially
change its local values and still be ignored or compensated before action.

【具体做法】

- Evaluated baseline, forced-zero and camera-preserving shuffled coordinates on
  32 samples/4 batches with identical sampling noise.
- Verified patched baseline matched ordinary baseline exactly.
- Measured raw-reader value delta, action delta and target MSE change with
  confidence intervals.
- Limited the conclusion to the coordinate path because Flow still affected
  confidence/detail-gate side paths.

【观察到的结果】

- Reader value delta was substantial (`0.353` for zero, `0.0503` for shuffle).
- Action delta RMSE was only about `0.000699`.
- Zero changed target MSE by `-0.0294%`; shuffle by `+0.0327%`.
- Both 95% intervals crossed zero.
- The checkpoint's Flow displacement and center separation were near zero with
  high entropy and weak seed reliability.

【当时的解释】

V98 numerically responded to Flow inside the reader, but sample-specific Flow
coordinates supplied no measurable positive action value. Downstream paths
ignored or compensated the address.

【后来怎么看（截至冻结终点）】

This made V99's nonduplicate fallback and observable-motion design necessary,
but not sufficient; V99 still needed its own downstream intervention.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；strong causal negative result within the stated coordinate-path scope.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 730-736。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

CI-0078, causal utility versus local sensitivity and downstream compensation.

### CI-0106 — V99 had positive same-epoch evidence but only partial closure

【时间 / 版本】

V99 epoch one and partial epoch two, 2026-07-22.

【当时在解决什么问题】

The user supplied V96, V98 and V99 logs to determine whether V99's repair helped
without introducing another regression.

【当时的 insight / hypothesis】

Compare complete versions at aligned epochs and separate producer
identifiability, reader address behavior, JEPA, action, gripper, world ownership
and execution. A necessary repair can remain incomplete downstream.

【具体做法】

- Compared epoch-one validation across all three runs rather than their unequal
  log tails.
- Followed moving/static/global gain and nonduplicate lane diagnostics.
- Compared action horizon, arm/gripper, future loss and role gradients.
- Kept V96's partial warm start as directional only.

【观察到的结果】

V99 epoch-one action RMSE (`0.10645`) was better than V98 (`0.10874`) and the
available V96 (`0.11148`), with lower first, tail and gripper RMSE. Moving gain
was positive, and partial epoch-two training had positive moving, static and
global gain. However future validation lagged V98, address entropy remained near
`0.993`, separation was small, world gradients stayed much weaker than policy,
gripper over-predicted events and execution interventions remained unfavorable.

【当时的解释】

V99 had repaired Flow identifiability without sacrificing early action quality.
It had not proved that the address reached action or that the dedicated world
middle owned future information.

【后来怎么看（截至冻结终点）】

The next design focused on downstream information ownership rather than making
the observable-motion loss stronger.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；producer repair supported, end-to-end closure unresolved.

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 737-740。
- 段级工件：见 Segment 011 boundary 的 `Primary historical artifacts`。

【关联】

V100 strict path, world/policy bypass and gripper/execution debt.

## Segment 011 synthesis

The history now contained both halves of a sound anti-shortcut argument:

```text
source audit:
  V98 lets prediction, motion support, confidence and fallback collapse together

causal audit:
  changing V98 coordinates changes reader values but not action quality
```

V99 replaced self-validating Flow with fixed observable evidence and a zero
baseline that applied only where the observation changed. Its early moving-region
gain and same-epoch action result were genuine positive evidence. But the
project correctly stopped short of calling it complete: correlation remained
diffuse, address separation small, world ownership weak and no V99-specific
zero/shuffle trace had yet shown that the repaired information survived to
action. The durable lesson was that anti-collapse is not “make the latent
non-zero”; it is “make correct, sample-aligned information outperform its
neutral and shuffled counterfactuals at the consumer.”

## Segment 012 — V100 strict ownership and V101 information balance

### Segment boundary

```text
Date:                 2026-07-22 to 2026-07-23
Visible messages:     741-802
Starts with:          deciding how to repair V99's incomplete causal chain
Ends with:            corrected full-log comparison of V96/V98/V99/V100
Primary historical artifacts:
  policy_runtime_v39.py
  flow_dino_evidence.py
  trunk_primitives.py
  trunk.py
  scripts/current_v100_strict_complementary_flow_jepa.sh
  scripts/current_v101_information_balanced_long_horizon.sh
  v96_bs8_formal.log
  v98_bs8_formal.log
  v99_observable_raw_flow_332_jepa.log
  v100.log
```

### CI-0107 — V100 repaired ownership with mechanism constraints, not stronger gates

【时间 / 版本】

V99 to V100, 2026-07-22.

【当时在解决什么问题】

V99 had made observable flow identifiable, but world gradients were weak,
policy and bottom could reread visual evidence, and the flow-local and fallback
lanes could still compete instead of carrying complementary information.

【当时的 insight / hypothesis】

The repair should make the intended path structurally necessary while retaining
bandwidth. It should not force nonzero flow, add an entropy target or let a gate
delete optional detail. A low-frequency base and flow-addressed high-frequency
residual are complementary rather than rival explanations.

【具体做法】

- Replaced the inherited Stage2 layer-wise LR ladder with the base LR for a
  scratch, single-stage run.
- Enforced `grounding -> world -> policy -> decoder`: policy stopped directly
  cross-attending visual tokens and the final decoder stopped rereading raw
  visual evidence or visual intent.
- Retained a complete low-frequency base and sampled only high-frequency
  residual with flow, then fused it without a learned deletion gate.
- Added an identity no-harm condition for static regions rather than a nonzero
  flow requirement.
- Made the future-change objective participate in optimization instead of
  remaining detached diagnostics.
- At the user's direction, deliberately deferred the proposed gripper closure
  and execution-controller shadow changes.

【观察到的结果】

All eight top blocks received action gradients, policy visual cross-attention
was absent, all three world blocks read the fused evidence, and flow-addressed
detail had finite coordinate gradients. V100 kept V99 behavior behind separate
configuration switches. A later full run showed detail/base energy remained
balanced and world/fine gradients became materially healthier.

【当时的解释】

This was a real producer-to-middle ownership repair. It established a stricter
route without claiming that the route had acquired action utility.

【后来怎么看（截至冻结终点）】

Partly successful. Full logs later showed better structural gradients and change
prediction but severe near-action bias. A mandatory path can still carry little
decision-relevant long-horizon information.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；ownership repair supported, end-to-end utility incomplete.

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 741-754 and 799-802。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。

【关联】

CI-0105, V98 causal intervention; V101 long-horizon balancing.

### CI-0108 — A path-depth edit exposed a selector/value identity mismatch

【时间 / 版本】

V100 implementation, 2026-07-22.

【当时在解决什么问题】

The decoder was reduced to the final two policy contracts as part of removing
visual bypasses.

【当时的 insight / hypothesis】

Changing the selector depth does not automatically change every parallel value
carrier. A locally plausible ownership edit can leave an adjacent tensor on the
old depth identity and silently change semantics or fail by shape.

【具体做法】

A directed end-to-end forward/backward check examined both the selected layer
contracts and their clean value queries before accepting the strict route.

【观察到的结果】

The selector had two rows while clean value queries still had eight. The check
caught the inconsistency, the value side was truncated to the same terminal two
contracts, and the action backward was rerun across all eight upstream blocks.

【当时的解释】

This was not evidence against the strict path. It was evidence that parallel
identity-bearing axes must be migrated together and verified at the consumer.

【后来怎么看（截至冻结终点）】

Retained as a concrete example of why shape checks, complete downstream mapping
and full-path backward tests are all necessary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 749-753。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。

【关联】

Mandatory subsystem-familiarity rule; later axis-identity audits.

### CI-0109 — Sparse observable motion creates a dataset-level zero-flow incentive

【时间 / 版本】

V96-V100 historical probe review, 2026-07-22.

【当时在解决什么问题】

The project needed to distinguish an architectural collapse from a dataset in
which most local observations are genuinely static.

【当时的 insight / hypothesis】

Global warp averages structurally favor identity motion when changed patches are
sparse. The correct response is conditional evidence and matched baselines, not
forcing all regions to move.

【具体做法】

- Measured nonzero patch displacement at multiple frame offsets.
- Compared oracle flow against identity globally and in high-motion regions.
- Interpreted learned-flow gains separately for moving and static supports.

【观察到的结果】

Approximate nonzero-patch rates were `8.4%`, `16.3%`, `27.2%`, `35.7%` and
`43.6%` at offsets `+1/+4/+12/+24/+48`. Oracle flow reduced warp error by about
`21%` at `+12/+24`, and by roughly `23-32%` in high-motion regions; the gain did
not keep increasing at `+48`.

【当时的解释】

Motion existed and was useful, especially at intermediate horizons, but global
static support could drown it. This justified observable-motion support and an
identity comparison while rejecting a blanket nonzero-flow objective.

【后来怎么看（截至冻结终点）】

Retained. It is an important instance of a hypothesis being narrowed by a data
probe rather than amplified from a single training metric.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；数据探针
- 对话：visible messages 755-757。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`+1/+4/+12/+24/+48`、`+12/+24`。

【关联】

CI-0099, correction of the earlier over-strong near-zero-flow interpretation.

### CI-0110 — Validation plateau was a distribution-and-path problem, not stopped optimization

【时间 / 版本】

V96-V100 retrospective and V101 design, 2026-07-22 to 2026-07-23.

【当时在解决什么问题】

Training flow and action losses kept improving while validation action and JEPA
flattened early, especially at long horizons.

【当时的 insight / hypothesis】

The model rapidly learned the easy, shared part of overlapping windows and then
spent capacity on static regions, near actions and objectives that a bypass
could compensate. Ordinary overfitting was too coarse an explanation.

【具体做法】

- Audited `stride=1` windows and found adjacent samples shared 23 of 24 actions.
- Designed a deterministic batch mixture with an unbiased half, motion-rich
  windows and a small event stratum; all-static data fell back to normal shuffle.
- Split action supervision into `1-4 / 5-12 / 13-24` bands with mild tail
  protection rather than an extreme reweighting.
- Normalized each JEPA horizon independently before equal aggregation.
- Kept future teacher information restricted to loss-position selection.

【观察到的结果】

The design was implemented as V101 and passed focused sampler, horizon-gradient
and teacher-isolation tests. At this point it had not yet produced validation
evidence.

【当时的解释】

Reweighting alone could not correct highly redundant minibatches, and a sampler
alone could not repair a bypassed consumer. Dataset exposure, loss allocation
and ownership had to be addressed together.

【后来怎么看（截至冻结终点）】

Partly retained. V101 later produced balanced JEPA horizons but poor deployed
tail/action behavior, and several sampler/loss/ownership changes were bundled.
Those logs rejected a broad success claim but could not isolate whether the
sampler itself improved generalization or introduced a distribution shift.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：未闭环
- 说明：尚未验证；V101 aggregate behavior could not isolate the sampler.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 758-766 and 770-778。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1-4 / 5-12 / 13-24`。

【关联】

Long-horizon action bands, JEPA horizon identity and sampling distribution.

### CI-0111 — V101 review found three residual aliases and bottlenecks after V100

【时间 / 版本】

V101 source review, 2026-07-22 to 2026-07-23.

【当时在解决什么问题】

V100's declared strict path still needed a second producer-to-consumer audit for
paths that were not obvious cross-attention edges.

【当时的 insight / hypothesis】

An information bypass can survive through modulation, an alias token or a fixed
residual coefficient even after the obvious reader has been removed.

【具体做法】

- Found strict policy modulation still reading `visual_memory.mean()` and
  replaced it with upstream world/trajectory summaries.
- Found the policy workspace entering Evidence MMDiT through a fixed `0.10`
  factor while terminal contracts omitted trajectory; replaced that artificial
  narrow ingress with fixed-variance typed fusion.
- Dropped action history simultaneously from direct history, history memory and
  the history-derived proposal alias.
- Audited the raw-flow coordinate frame after token fusion.

【观察到的结果】

The coordinate audit found source-aligned flow detail had been added to the
latest-frame DINO grid, although moving objects made those cells semantically
different. V101 instead fused it with the source DINO chart and retained the
latest chart separately. Focused and broad tests passed, and optimizer ownership
was checked after backward connectivity.

【当时的解释】

The initial V100 path repair was incomplete in exactly the ways a local review
misses: modulation, residual scale, aliases and coordinate identity. These were
not reasons to add more losses; they were reasons to finish the dataflow map.

【后来怎么看（截至冻结终点）】

Retained. It anticipates the later mandatory subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 767-778。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`visual_memory.mean()`。

【关联】

CI-0108, parallel-axis migration; all later bypass audits.

### CI-0112 — Historical log identity includes location and completeness, not only filename

【时间 / 版本】

V96-V100 audit correction, 2026-07-23.

【当时在解决什么问题】

The user asked for four complete local logs. The first review instead read four
attachment-cache snapshots, two of which were different captures of one V98 run,
then treated their truncated tails as experiment endpoints.

【当时的 insight / hypothesis】

Before comparing epochs, an audit must establish the exact requested storage
location, run identity, configured epoch count, terminal record and whether a
file is a live log, immutable upload or partial snapshot.

【具体做法】

After the user corrected the source, the workspace root was scanned and the
actual `v96_bs8_formal.log`, `v98_bs8_formal.log`, V99 log and `v100.log` were
found. Their complete eight-epoch trajectories were then parsed.

【观察到的结果】

Each local log contained 1,136 training records through epoch eight, whereas the
attachment copies ended at different intermediate batches. The earlier
attachment-based conclusion was explicitly discarded.

【当时的解释】

This was an audit-process failure, not model uncertainty. The user's frustration
correctly tightened the evidence boundary: do not substitute a convenient
artifact for the one named in the request.

【后来怎么看（截至冻结终点）】

Retained as both an evidence-provenance rule and an interaction lesson. The
correction mattered more than defending the earlier analysis.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；process correction.

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 782-800, especially user corrections 787, 792 and 794。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`v96_bs8_formal.log`、`v98_bs8_formal.log`、`v100.log`。

【关联】

CI-0001, realized experiment identity; later source/log provenance failures.

### CI-0113 — V96-V100 separated numerical baselines from mechanism baselines

【时间 / 版本】

Complete eight-epoch V96/V98/V99/V100 logs, reviewed 2026-07-23.

【当时在解决什么问题】

The project needed a full-trajectory comparison rather than another early-epoch
or single-metric verdict.

【当时的 insight / hypothesis】

The version with the best action number need not contain the healthiest intended
mechanism. Numerical, observability, ownership and long-horizon baselines must be
kept separate.

【具体做法】

Compared eight-epoch training flow, JEPA, validation full/first/tail action,
gripper events, address entropy and separation, role gradients, loss composition,
proposal and execution ablations.

【观察到的结果】

- V98 had the best final action RMSE (`0.07966`) and tail (`0.09346`) but raw
  flow collapsed toward zero and prior action interventions showed no utility.
- V99 maintained nonzero observable flow and positive moving/static warp gain,
  yet its address stayed diffuse and final action (`0.08672`) was worse.
- V100 balanced detail/base energy, strengthened world and fine gradients and
  improved change prediction, but drove first RMSE to `0.00998` while tail
  remained `0.09740`, a tail/first ratio of about `9.76`.
- V96 remained a useful older numerical reference but had an incomplete Stage1
  warm start and therefore was not a clean structural comparator.
- Across all runs, training objectives kept descending after validation and
  long-horizon generalization had largely plateaued.

【当时的解释】

V98 was the action-number baseline, V99 the flow-observability baseline and V100
the ownership-gradient baseline. None closed `correct flow -> world change ->
long-horizon action`.

【后来怎么看（截至冻结终点）】

Retained. This multi-anchor discipline became increasingly important in later
schema comparisons.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 798-802。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。

【关联】

V101 validation and later V120/V25 behavioral anchors.

### CI-0114 — Non-finite diagnostics must be traced to their computational role

【时间 / 版本】

V100 full-log audit, 2026-07-23.

【当时在解决什么问题】

The log contained 1,144 `+inf` values and an audit tool initially labeled the run
as a critical numerical failure.

【当时的 insight / hypothesis】

A non-finite audit field is not automatically a non-finite model. One must trace
whether it reaches a loss, output, gradient, optimizer state or only detached
logging.

【具体做法】

All non-finite fields were enumerated and the corresponding concentration metric
was traced to its source calculation.

【观察到的结果】

Every `+inf` belonged to `detail_address_concentration`. Invalid candidates had
been filled with the floating-point minimum and then included in
`max(logits)-mean(logits)`, causing overflow. The value was detached and never
entered loss or backward; all model losses and gradients remained finite.

【当时的解释】

The run was not numerically exploding. The diagnostic needed to average only
valid candidates, and the audit severity classification needed computational
provenance.

【后来怎么看（截至冻结终点）】

Retained. Later gradient sentinels and compact log audits formalized the same
distinction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 798-800。
- 段级工件：见 Segment 012 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`detail_address_concentration`。

【关联】

Non-finite parameter sentinel and diagnostic semantics.

## Segment 012 synthesis

V100 proved that a stricter, more balanced representation route could be built
without forcing flow or adding a deletion gate. Its full run also proved why
that was insufficient: healthier world gradients and future-change prediction
coexisted with an extreme first-step improvement and a stagnant tail. V101 was
therefore designed around exposure, horizon identity and the remaining hidden
bypasses rather than another stronger flow loss. Just as importantly, the log
review itself failed once by substituting attachment snapshots for the complete
local files the user named. Correcting that mistake established a durable audit
rule: artifact provenance and completeness must be proven before any trajectory
interpretation, and a technically polished analysis of the wrong files is still
wrong.

## Segment 013 — V101 causal localization and V102 late factual read

### Segment boundary

```text
Date:                 2026-07-23
Visible messages:     803-877
Starts with:          V101 early training compared with V95/V96/V98
Ends with:            completed V102 anchor-world/late-detail implementation audit
Primary historical artifacts:
  V101 training logs through epoch two
  V101 action-path probe schemas v1-v3
  run_v101_action_path_probe.sh
  flow_dino_evidence.py
  trunk.py
  time_domain_mmdit.py
  scripts/current_v102_anchor_world_late_raw_detail.sh
```

### CI-0115 — V101 made policy ingress dominant while spatial address stayed trainable but starved

【时间 / 版本】

V101 early epoch one and epoch two, 2026-07-23.

【当时在解决什么问题】

The V101 log showed persistent clipping, strong policy gradients and weak
fine-address gradients. The user explicitly required separate causal accounts
unless evidence linked them.

【当时的 insight / hypothesis】

Forward energy, coordinate ownership and gradient strength are different facts.
The address path can carry a large raw-detail tensor while the flow-specific
choice of position receives almost no useful optimization signal.

【具体做法】

- Traced detach, gating, reader placement, all downstream blocks and loss owners.
- Compared aligned early gradients from V98, V100 and V101 before and after the
  global clip.
- Converted cell separation and candidate radius into high-grid pixel geometry.
- Separated the direct cause of the large global norm from its secondary effect
  on a weak branch.

【观察到的结果】

- The V101 address reader had about 46 times less preclip gradient than V98 and
  about 82 times less after clipping; it was weak but still above AdamW epsilon.
- The detail router's aggregate RMS update was near or below optimizer epsilon
  early in training.
- Raw detail still carried roughly 40-48% of local fused energy; it had not been
  deleted.
- Address separation was only about `0.06-0.11` reader cell while attention
  could search 49 candidates over a much wider neighborhood, so local attention
  could compensate for a nearly identity flow center.
- Changing policy-workspace ingress from `0.10` to fixed-variance `0.707`
  increased policy and global gradients. Global clipping further reduced the
  address update but did not cause the original V98-to-V100 collapse in address
  gradient.

【当时的解释】

The primary address weakness came from late placement, a long marginal-value
path and compensable search geometry. Policy dominance was a separate direct
effect of the stronger workspace ingress; common clipping only made the address
starvation worse.

【后来怎么看（截至冻结终点）】

Retained. This is a strong example of refusing to manufacture one cause from two
correlated metrics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 803-821 and 827-830。
- 段级工件：见 Segment 013 boundary 的 `Primary historical artifacts`。

【关联】

CI-0111, V101 fixed-variance ingress; action-path causal probe.

### CI-0116 — Balanced future representation did not imply balanced deployed action

【时间 / 版本】

V101 epoch-one validation and partial epoch two, 2026-07-23.

【当时在解决什么问题】

V101 was intended to protect long-horizon training, but its first validation was
worse than earlier versions despite improving the first action.

【当时的 insight / hypothesis】

Teacher-forced horizon balance can succeed while the deployment sampler or the
representation-to-action handoff remains highly near-biased.

【具体做法】

Compared epoch-one V95/V96/V98/V101 at full, first, first-eight, tail and
`1-4/5-12/13-24`, while separately reading JEPA horizons and epoch-two training
bands.

【观察到的结果】

V101 epoch-one full RMSE was `0.12028`; first was `0.04494`, but tail was
`0.13974` and `13-24` was `0.14989`. JEPA h4-to-h48 error expanded only about
4.8%, whereas deployed `13-24` action error was about 2.9 times `1-4`.
Epoch-two teacher-forced bands and router gradients improved without yet proving
deployment recovery.

【当时的解释】

The largest fault domain moved downstream of future representation supervision:
world/JEPA information lacked identifiable action ownership, while the strong
policy workspace could solve current and near action without it.

【后来怎么看（截至冻结终点）】

Retained as localization, not a unique line-level diagnosis. It motivated a
matched action-path intervention rather than another training loss.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；过程/决策记录
- 对话：visible messages 822-830。
- 段级工件：见 Segment 013 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1-4/5-12/13-24`。

【关联】

Long-horizon deployment, five-step sampler and policy-workspace ownership.

### CI-0117 — A causal probe must preserve the identity it claims to scramble

【时间 / 版本】

V101 action-path probe schemas v1-v3, 2026-07-23.

【当时在解决什么问题】

The first probe compared zero and shuffle interventions for policy workspace,
world output and raw-flow address under matched data and sampling noise.

【当时的 insight / hypothesis】

An intervention is only meaningful if it destroys the tested semantics while
preserving unrelated identities. Reordering complete tokens together with their
positional encoding can be a legal set permutation, and moving time and space at
once cannot attribute either dimension.

【具体做法】

- Kept checkpoint, batch, proposal, initial noise and five-step inference fixed.
- Replaced whole-world token shuffling with interventions on only the residual
  written after grounding, preserving slots and positional encodings.
- Split world residual mismatch into anchor and spatial shuffles.
- Defined raw detail after the output projection as `full reader - base-only`,
  then zeroed or spatially shuffled that exact residual.
- Stratified batches across all five validation episodes and event windows and
  used episode-cluster confidence intervals.

【观察到的结果】

The probe evolved through two semantic corrections before schema v3. The final
run covered 80 samples, all five episodes, 28 true gripper events and 11 modes;
patched baseline identity error was exactly zero.

【当时的解释】

The earlier “world shuffle has no effect” reading was too strong because the
intervention did not break the intended relation cleanly. Correct probe algebra
was part of the model audit, not merely evaluation plumbing.

【后来怎么看（截至冻结终点）】

Retained. It became a durable rule for K, camera, interval and lane
interventions in later schemas.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 831-864。
- 段级工件：见 Segment 013 boundary 的 `Primary historical artifacts`。

【关联】

Matched interventions, axis identity and causal audit design.

### CI-0118 — V101 used world time, was harmed by world space, and ignored projected high-frequency detail

【时间 / 版本】

V101 epoch-two action-path probe v3, 2026-07-23.

【当时在解决什么问题】

The project needed to know whether world and late spatial evidence were globally
weak or contained opposing useful and harmful components.

【当时的 insight / hypothesis】

Aggregate zero ablation can hide cancellation. Time/phase ownership, spatial
binding and factual detail must be intervened separately.

【具体做法】

Ran residual zero, anchor shuffle, spatial shuffle, joint shuffle, raw-detail
zero/shuffle and raw-flow address zero/shuffle on the same checkpoint and
samples.

【观察到的结果】

- World residual zero changed MSE by about `+1.22%` with an interval crossing
  zero.
- Anchor shuffle worsened MSE by about `+2.10%`; the near band worsened most.
- Spatial shuffle improved MSE by about `1.00%`, indicating harmful xy binding.
- Raw-detail residual had large selector/value norms (`9.11/15.41`) and was
  strongly changed by the probe, yet zero or shuffle altered action MSE by only
  about `0.064%/0.044%`, both uncertain.
- Policy temporal shuffle worsened MSE by about `133%`, confirming policy as the
  dominant route.
- None of 36 predicted gripper events matched 28 true events within tolerance.

【当时的解释】

World was not simply weak: useful anchor organization and harmful spatial writes
cancelled. High-frequency detail existed locally but was not used by deployed
action. Policy stabilized the trajectory but did not learn event timing.

【后来怎么看（截至冻结终点）】

Retained as strong causal evidence within the tested checkpoint. It directly
motivated preserving world phase while postponing precise factual reads.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；strong matched-intervention result.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 839-865。
- 段级工件：见 Segment 013 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.064%/0.044%`。

【关联】

V102 anchor-world/late-detail design and later factual/intent separation.

### CI-0119 — V102 delayed high-resolution facts until phase context existed

【时间 / 版本】

V102 design and implementation, 2026-07-23.

【当时在解决什么问题】

V101 passed high-frequency detail through the entire grounding/world chain even
though world spatial writing was harmful and the detail was ignored at action.

【当时的 insight / hypothesis】

World should organize phase/anchor and camera context; a bounded, query-specific
high-resolution read should occur afterward, before policy, without reopening a
complete raw visual bypass.

【具体做法】

- Restricted world writers to anchor/camera residuals and removed xy-specific
  writes.
- Compiled observation-only raw/flow detail and read it at the final
  world-to-policy boundary, separately per camera and aligned by horizon and
  action basis.
- Used a fixed `0.25` additive update with exact zero semantics and no learned
  amplitude deletion gate.
- Fixed a pre-existing decoder error that flattened `24 x basis` as a 96-step
  sequence and interpolated it back to 24, mixing time and basis identities.
- Prevented `refine_raw_evidence` from mutating and consuming shared raw context
  across preview, counterfactual and five-step forwards.

【观察到的结果】

Three static-review rounds caught and fixed an old-version return-arity break,
dead V102-only parameters still owned by the optimizer, incomplete same-shape
resume identity and cache lifecycle risks. Directed PyTorch, parser, shell and
compatibility tests passed before smoke was authorized.

【当时的解释】

The repair conserved information while moving its consumer to a point with
better action-phase context. It did not claim experimental gain before a run.

【后来怎么看（截至冻结终点）】

Later complete V102 logs did not validate this realization: raw-detail share
fell from about `0.218` to `0.038`, late-detail/trajectory stayed near `0.001`,
and middle/far action, JEPA and gripper ended weaker than V101. The late-read
ownership principle and the repeated reverse-review process remained valuable,
but V102's carrier had collapsed before useful action consumption.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；late-read principle retained, V102 realization behaviorally rejected.

【证据与锚点】

- 类型：测试/运行观测；因果干预；方案/设计记录；推断/未决
- 对话：visible messages 866-877。
- 段级工件：见 Segment 013 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`refine_raw_evidence`。

【关联】

Late factual reading, camera ownership, cache lifecycle and checkpoint identity.

## Segment 013 synthesis

V101 supplied the first sufficiently discriminating action-path probe in this
part of the history. Its successive corrections were as important as its final
numbers: a shuffle that carries its position with it may not destroy spatial
meaning, and a joint time-space shuffle cannot localize either axis. Once the
probe respected identity, it showed three different realities behind one weak
aggregate: world phase was useful, world xy writing was harmful, and projected
high-frequency detail was locally strong but action-irrelevant. V102 followed
that evidence by preserving phase ownership and delaying precise factual read.
The implementation then required three review passes to catch arity, cache,
optimizer and resume-contract failures—early evidence that “the intended graph
looks right” is not a sufficient closure criterion.

## Segment 014 — Joint address ownership, executable contracts and audit-method reset

### Segment boundary

```text
Date:                 2026-07-23 to 2026-07-26
Visible messages:     878-945
Starts with:          later V101/V102 evidence and possible AttnRes use
Ends with:            choosing V96 as a clean reconstruction point and issue-ledger workflow
Primary historical artifacts:
  v101_long_run_01.log
  v102_real_work.log
  flow_dino_dataset_motion_probe_v1
  attnres_332_migration_plan.md
  soft_multires_address_lattice.md
  00_CURRENT_ARCHITECTURE_CONTRACT.md
```

### CI-0120 — AttnRes was proposed for organized-source selection, not spatial addressing

【时间 / 版本】

V101-V102 design discussion, 2026-07-23.

【当时在解决什么问题】

Information formed in G/W appeared to be diluted before action, prompting the
user to ask whether Kimi Attention Residuals could help near the 3-3-2 hierarchy.

【当时的 insight / hypothesis】

Two kinds of selection must remain separate: spatial selection answers where to
read; residual/source selection answers which already-organized evidence or
depth delta to use. AttnRes is suited to the latter.

【具体做法】

- Scoped candidate use to the `3->3`, `3->2`, top-policy-to-bottom and Evidence
  organizer boundaries rather than replacing every block residual.
- Kept flow-conditioned high-resolution attention as a protected source.
- Proposed per-query selection of G/W depth increments after their owners had
  formed, with main state outside the competition.

【观察到的结果】

The discussion produced a prospective migration design, not a mainline code
change. V102 early logs made the `world->policy` boundary the most plausible
first study point because late detail had gradient but diminishing trajectory
write.

【当时的解释】

AttnRes could improve organization and downstream access; it could not make
flow accurate, create missing phase supervision or replace spatial matching.

【后来怎么看（截至冻结终点）】

Retained as bounded prospective research. Later implementations must be judged
from active source rather than inferred from this plan.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：未闭环
- 说明：尚未验证 / prospective.

【证据与锚点】

- 类型：日志观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 878-891 and 921-928。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

Typed G/W/P depth routing and later RoleDeltaAttnRes.

### CI-0121 — Final spatial address was reassigned from flow alone to a joint posterior

【时间 / 版本】

Post-V102 address design, 2026-07-23.

【当时在解决什么问题】

The analysis had implicitly blamed a weak learned flow for not producing a
precise final address. The user challenged that ownership assumption.

【当时的 insight / hypothesis】

Flow should provide a geometric proposal, not own final address. DINO/raw
matching, JEPA future hypothesis and the early G blocks should jointly organize
a soft spatial posterior.

【具体做法】

The intended G roles were articulated as alignment, flow correction and common
coordinate organization. Each coarse cell would retain multiple autonomous,
continuous high-resolution address slots over broader candidates rather than a
hard one-to-one pointer or one early weighted value.

【观察到的结果】

Source review showed V102 implemented role permissions but not this functional
G1/G2/G3 address organization: late-detail query construction had removed the
grounding semantic query, leaving flow/base geometry with too much ownership.

【当时的解释】

The failure was not simply “learned flow is weak.” The graph assigned the wrong
module final-address responsibility and compressed alternatives before the
joint semantic/action query existed.

【后来怎么看（截至冻结终点）】

Retained as a principle and later absorbed by progressive/joint-posterior
grounding designs. The exact V102/V103 formulation was superseded rather than
validated as an independent recovery unit.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；joint-posterior 原则保留，早期 formulation 已被替代。

【证据与锚点】

- 类型：源码事实；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 892-902。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

CI-0115, compensable local search; V103 soft address bank.

### CI-0122 — Structural availability was defined as six separate obligations

【时间 / 版本】

Architecture-design reset, 2026-07-23.

【当时在解决什么问题】

The user clarified that the immediate goal was to remove architecture-imposed
limits while treating data limitations separately.

【当时的 insight / hypothesis】

The architecture need not force use of every source, but it must keep valuable
information accessible, selectable, differentiable and transmissible without a
shorter route making it irrelevant.

【具体做法】

Six obligations were recorded: avoid early information loss; do not assign flow
exclusive address ownership; make 3-3-2 functional division rather than
isolation; preserve typed deltas and a main state; keep optional selection
zero-safe without pooled shortcuts; and require consumer-level causal tests.

【观察到的结果】

The new soft-address and residual-selection ideas could address structural
compression and organization, but explicitly could not solve data coverage,
true long-memory supervision, gripper timing or execution-controller utility.

【当时的解释】

This separated “mechanism has no structural handicap” from “the task is
statistically identifiable,” preventing a new architecture from being sold as a
universal fix.

【后来怎么看（截至冻结终点）】

Retained as a useful boundary, later strengthened by producer-consumer and
consumer-producer audits.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录
- 对话：visible messages 903-912。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

Information conservation, anti-shortcut boundaries and current subsystem rule.

### CI-0123 — V102 full logs showed carrier collapse, not uniformly weak world learning

【时间 / 版本】

Complete V101/V102 eight-epoch comparison, 2026-07-26.

【当时在解决什么问题】

The project needed to determine whether V102's late-detail repair improved the
full run and whether AttnRes targeted its actual failure.

【当时的 insight / hypothesis】

A route can gain gradients over training while its identifiable carrier loses
relative bandwidth. Aggregate world strength cannot substitute for tracking the
specific factual residual that must survive.

【具体做法】

After correcting the artifact source, the complete local logs were aligned over
all eight epochs and compared at best and final validation, role gradients,
raw-detail share, late-detail/trajectory ratio, JEPA, middle/far action and
gripper.

【观察到的结果】

Both runs were best near epoch six and regressed in epochs seven-eight. V102's
policy/world gradient ratio improved from about `30.5x` to `4.9x`, so world was
not uniformly disconnected. But raw-detail share fell from about `0.218` to
`0.038`, late-detail/trajectory stayed around `0.001`, and V102 ended weaker in
middle/far action, JEPA and gripper than V101.

【当时的解释】

V102's failure was loss of high-frequency carrier capacity and action ownership,
not simply insufficient world training. AttnRes might retrieve depth information
but could not restore a carrier already compressed to near irrelevance.

【后来怎么看（截至冻结终点）】

Retained. It sharpened the distinction between block utilization and information
utilization.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 916-919 and corrected full-log review 924-933。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

V102 late factual path, V103 multihypothesis bank and later bandwidth audits.

### CI-0124 — Repeatedly reading partial attachments damaged both conclusions and trust

【时间 / 版本】

V101/V102 full-log correction, 2026-07-26.

【当时在解决什么问题】

Despite an earlier correction in the same conversation, the review again used a
partial uploaded V102 text instead of the complete local logs the user had named.

【当时的 insight / hypothesis】

An evidence-provenance rule that is only stated but not executed is not a
process repair. Repetition changes the interaction boundary: the user can no
longer reasonably trust assurances of completeness without visible artifact
verification.

【具体做法】

The user said the runs were complete and that the files had not been read
carefully. The assistant withdrew the incomplete-run claims, located both local
eight-epoch logs, stated their record counts, and redid the comparison.

【观察到的结果】

The corrected evidence reversed important judgments: V102 had eight validations,
world gradients strengthened late, and the persistent failure was the collapsing
detail carrier. The user's frustration then shifted the requested method from
more architectural prose to a full static issue ledger and executable checks.

【当时的解释】

The frustration had a concrete trigger and a technically valid effect on the
workflow. It is not itself evidence about V102, but it records why later claims
needed stronger provenance and less reassurance.

【后来怎么看（截至冻结终点）】

Retained with the earlier correction rather than sanitized away. Repeated error,
not the intensity of the wording, is the important process fact.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；interaction/process lesson.

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 924-933, especially user correction 929。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

CI-0112, first wrong-artifact correction; later trust breakdown.

### CI-0125 — V98-V102 accumulated defensive constraints faster than positive paths were validated

【时间 / 版本】

V98-V102 retrospective, 2026-07-26.

【当时在解决什么问题】

The user asked why each repair seemed to introduce more structural problems.

【当时的 insight / hypothesis】

Successive versions blocked newly discovered shortcuts before proving that the
remaining positive route had sufficient bandwidth, correct axes, useful query
semantics and consumer ownership. The model therefore migrated to the next
cheapest path.

【具体做法】

Replayed the intent and observed failure of V98 raw/3-3-2, V99 observable flow,
V100 strict ownership, V101 long-horizon balance/strong workspace and V102
anchor-only world/late detail. Also inspected wrapper inheritance and mutable
shared source.

【观察到的结果】

Version scripts froze parameter combinations but not unique computation graphs;
many feature switches and continuously changing shared source made ancestry
ambiguous. Each version improved a local property while exposing or creating a
new bottleneck: zero-flow avoidance without action use, strict route without
long-horizon bandwidth, strong workspace dominance, then detail-carrier collapse.

【当时的解释】

The problem was not that anti-shortcut constraints are inherently wrong. They
were applied without first validating the capacity and semantics of the route
left behind, and multiple boundaries changed per version.

【后来怎么看（截至冻结终点）】

Retained. Later replay ledgers separate regression anchors, performance anchors
and structural-health anchors for this reason.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录；历史检索
- 对话：visible messages 934-938。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

Version identity, positive-path closure and inherited regression amplification.

### CI-0126 — Aspirational contracts and local static review were explicitly rejected

【时间 / 版本】

Method reset after V102, 2026-07-26.

【当时在解决什么问题】

The user objected that several contracts had already been written while obvious
static structural errors survived.

【当时的 insight / hypothesis】

A contract is useful only when it describes one realized graph and its claims are
enforced by executable boundary tests. Shape, finiteness, nonzero gradients,
comments and named roles do not establish functional closure.

【具体做法】

- Acknowledged that prior review asked whether each local patch solved its target
  rather than retracing input-to-action-to-loss end to end.
- Identified static contradictions that should have blocked release: zeroing the
  grounding query in V102, giving bottom only terminal P contracts, and strongly
  normalizing policy workspace over weaker upstream carriers.
- Chose V96 as a relatively clean audit/reconstruction point, without declaring
  it correct or using V98's good RMSE as structural proof.
- Adopted a two-stage workflow: first record every model-path issue and its
  evidence/dependencies, then solve them in order.

【观察到的结果】

The user's angry description of the contracts directly narrowed the review
standard. The assistant stopped proposing another renamed document as the fix
and accepted responsibility for errors that static dataflow review should have
caught.

【当时的解释】

This was a justified process correction. Emotional intensity was not used as
technical proof, but the trigger—repeatedly passing structurally contradictory
graphs—was real and changed the audit method.

【后来怎么看（截至冻结终点）】

Retained. It is direct ancestry for the current mandatory subsystem-familiarity
rule and pure-issue-ledger practice.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；method and interaction turning point.

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 939-945。
- 段级工件：见 Segment 014 boundary 的 `Primary historical artifacts`。

【关联】

Current `AGENTS.md`, producer-to-consumer and reverse-consumer audit.

## Segment 014 synthesis

This segment changed both architectural ownership and the review process. The
user corrected the assumption that learned flow should own final address; the
result was a joint soft posterior over retained high-resolution hypotheses,
organized by DINO/raw/JEPA and G queries. Complete V102 evidence then showed why
that mattered: world gradients could recover while the factual carrier itself
collapsed. A second wrong-artifact review, followed by the user's sharper
criticism of repeated “contracts,” made the procedural failure undeniable. The
durable outcome was not another document name. It was the requirement to freeze
the realized graph, enumerate static contradictions before editing, conserve
positive-path bandwidth when closing shortcuts, and verify every ownership
claim at its actual consumer.

## Segment 015 — V103 reconstruction and repeated reverse-completion audits

### Segment boundary

```text
Date:                 2026-07-26
Visible messages:     946-1098
Starts with:          V96-rooted model-only issue audit
Ends with:            V103 launcher inheritance failure caught before training
Primary historical artifacts:
  00_CURRENT_ARCHITECTURE_CONTRACT.md issue set M01-M26
  flow_dino_evidence.py
  trunk.py
  time_domain_mmdit.py
  V103 model-path probe
  scripts/current_v103_typed_predictive_flow_jepa.sh
```

### CI-0127 — V103 retained address hypotheses until action-conditioned consumption

【时间 / 版本】

V103 reconstruction, 2026-07-26.

【当时在解决什么问题】

V96-V102 repeatedly compressed candidate positions and high-frequency facts
before a semantic, temporal and action-basis query could decide which one mattered.

【当时的 insight / hypothesis】

An observation can be compiled once without being reduced to one value. Address
hypotheses and local values should remain observation-owned, while final
selection is conditioned at the world/policy consumer.

【具体做法】

- Added an observation-only soft lattice retaining camera, coarse cell, multiple
  slots and multiple fine candidates.
- Kept flow as one geometric expert and permitted DINO, source-target raw
  correspondence and world spatial state to correct it.
- Selected camera/slot/fine offset with horizon, basis and policy queries only at
  the late consumer.
- Required exact zero detail when fine values were zero and ordinary action/JEPA
  gradients back to raw, DINO and flow compilers.

【观察到的结果】

Local and full-policy forward/backward checks preserved multiple hypotheses,
compiled the observation bank once across ODE steps and delivered gradients to
the intended producers. This established structure but not trained utility.

【当时的解释】

The repair moved compression to the first consumer that possessed the necessary
action identity rather than making flow harder or adding an address loss.

【后来怎么看（截至冻结终点）】

Partly retained. Subsequent audits confirmed the downstream axis-loss warning,
but no matched V103 result established this exact retained-hypothesis
implementation as a healthy endpoint.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；轴保留原则得到后续支持，V103 realization 未获独立行为闭环。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 947-964。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

CI-0121, soft multiresolution address hypothesis.

### CI-0128 — V103 turned 3-3-2 labels into typed residual ownership

【时间 / 版本】

V103 reconstruction, 2026-07-26.

【当时在解决什么问题】

Prior 3-3-2 versions restricted write regions but did not reliably preserve what
each depth produced or let bottom retrieve it independently.

【当时的 insight / hypothesis】

The shared carrier should remain outside routing competition, while real G/W/P
block deltas retain their owner axes and are conditionally read at boundaries.

【具体做法】

- Stored G/W deltas by anchor and camera and P deltas by time and action basis.
- Introduced continuous low-rank G-to-W, W-to-P and P-to-MMDiT reads.
- Removed the fixed equal-variance policy-workspace superhighway.
- Preserved coarse rollout, typed deltas and protected high-frequency detail as
  semantically distinct bottom inputs instead of deleting all parallel evidence.
- Removed noisy action and proposal from G updates; action conditioning began at
  W, while G retained observation, goal, state and history alignment.

【观察到的结果】

Action loss reached all three role groups and boundary readers, with world xy
permissions and consumer identities checked explicitly. Coarse rollout was
retained after review showed it was complementary rather than a raw-detail bypass.

【当时的解释】

Role ownership is not synonymous with forcing one serial tensor. Distinct
semantic carriers can coexist if each has a defined consumer and intervention.

【后来怎么看（截至冻结终点）】

Retained as a stronger formulation than the earlier “strict single path.” Later
history repeatedly preserved typed ownership, while the exact V103 route
differentiation was not isolated before subsequent designs replaced it.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；typed ownership 保留，V103 identifiability 未被独立隔离。

【证据与锚点】

- 类型：因果干预；推断/未决
- 对话：visible messages 961-964, 982-1007 and 1013-1018。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Typed delta banks, coarse/fine complementarity and route-collapse probes.

### CI-0129 — Exact null and stateless phase were made query conditions rather than values

【时间 / 版本】

V103 language/history/phase repair, 2026-07-26.

【当时在解决什么问题】

Zeroed language or history could pass through biased projections and leave a
learned template; phase was absent from W organization and risked becoming an
unowned action-writing sidecar.

【当时的 insight / hypothesis】

A null condition must be exact in content space, and stateless phase should
shape queries into world/detail evidence rather than manufacture observation or
directly write actions.

【具体做法】

- Defined conditional contribution as resampled conditional output minus the
  corresponding null output, then applied keep/drop.
- Formed a soft phase belief from observable goal/history context.
- Gave W blocks independent phase/condition query projections and passed phase
  into W-to-P and high-resolution address selection.

【观察到的结果】

Different language/history inputs became bit-identical at all relevant outputs
when keep was zero; ordinary action loss reached the T5 resampler, phase
estimator, world and detail queries without a direct phase-to-action write.

【当时的解释】

This addressed hidden templates and missing query conditioning without using a
learned null as an optional-value shortcut.

【后来怎么看（截至冻结终点）】

Retained structurally; later trained logs would reveal strong goal-token
homogenization, so exact connectivity was not sufficient.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：测试/运行观测；推断/未决
- 对话：visible messages 965 and 983。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Language/phase identifiability and later learned-null anti-shortcut boundary.

### CI-0130 — JEPA context exclusion and future-delta prediction are distinct contracts

【时间 / 版本】

V103 JEPA repair, 2026-07-26.

【当时在解决什么问题】

The old mask was applied after frozen DINO had mixed the full frame, and the
future query could copy the last absolute DINO state while predicting an
absolute future dominated by static content.

【当时的 insight / hypothesis】

Input exclusion and target algebra solve different shortcuts. They must be fixed
at their own boundaries rather than hidden behind one auxiliary loss.

【具体做法】

- Applied masking to raw context before any trainable spatial mixing.
- Used one observation-derived mask for online context and all supervised future
  horizons during training; future teacher only supplied target values.
- Disabled masking at deployment.
- Predicted frozen-teacher future minus current-teacher delta instead of
  initializing from and primarily reconstructing the last absolute DINO value.
- Ensured the post-G address bank reused the same masked raw/DINO boundary so it
  could not reintroduce hidden appearance.

【观察到的结果】

Masked-pixel replacement left online encoding unchanged, correct delta produced
near-zero loss, zero delta received a corrective gradient, and changing future
teacher altered supervision but not the action velocity field.

【当时的解释】

The repair removed both a context leak and a static-copy target, while retaining
full observation for deployed action.

【后来怎么看（截至冻结终点）】

Retained, and later history confirmed the missing part: independent horizon
deltas did not by themselves create chronological/interval ownership. V106 and
the later S/Teacher/W line introduced explicit interval-stage and adjacent-future
semantics while preserving teacher isolation.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续 interval objectives 吸收；两类 shortcut 的边界仍保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 967-970 and 1008-1020。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Sequential `+4/+12/+24/+48` memory and teacher isolation.

### CI-0131 — Reverse audit found repeated downstream loss of supposedly retained axes

【时间 / 版本】

V103 repeated completion audit, 2026-07-26.

【当时在解决什么问题】

After the primary V103 implementation passed forward/backward tests, the user
required continued review rather than treating connectivity as completion.

【当时的 insight / hypothesis】

Every preserved identity must be traced all the way to its terminal consumer.
An upstream multi-axis representation is not conserved if a later helper averages,
truncates or repeatedly scales it.

【具体做法】

Repeated producer-to-consumer and consumer-to-producer passes found and repaired:

- action-basis detail averaged before MMDiT;
- the independent `+48` anchor truncated before W-to-P;
- fixed `0.10 x 0.10 x 0.25` scaling that reduced G evidence to `0.0025` while P
  entered at `0.25`;
- a nominal flow-prior floor that uncertainty could broaden into an uninformative
  constant;
- anchor-only W that could read but not write coarse xy change;
- W xy averaged again before high-resolution address selection;
- `source_raw` passed to the lattice but used only for a shape check;
- an independent context/target mask mismatch.

【观察到的结果】

Each issue had valid shapes and often nonzero gradients. Only end-to-end axis,
scale and semantic tracing exposed the contradiction. Fixes retained `+48` as a
far context rather than pretending it was action step 24, unified G/W/P bottom
read scale, restored coarse W spatial writing and used source-target raw only in
selector keys while values remained target high-pass facts.

【当时的解释】

This was concrete evidence that the earlier static-review standard was too local.
No single unit test could substitute for mapping transformations and consumers.

【后来怎么看（截至冻结终点）】

Retained as foundational ancestry of the present subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 981-1026。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`source_raw`。

【关联】

CI-0126 and current mandatory complete-boundary review.

### CI-0132 — Optimizer ownership and deployment-root reachability require exact sets

【时间 / 版本】

V103 completion audit, 2026-07-26.

【当时在解决什么问题】

Many legacy modules remained instantiated and optimizer-owned even though the
active V103 deployment graph or loss could never execute them.

【当时的 insight / hypothesis】

“All parameters are in an optimizer” is not the correct invariant. The exact
trainable set must equal active model parameters with a real loss owner, and the
deployment-root set may exclude only explicitly supervised auxiliary heads.

【具体做法】

- Backpropagated the real complete loss and enumerated `grad=None` by module.
- Removed or froze unreachable midcut heads, diagnostic G/W contract heads,
  disabled P rollout-cross layers, strict-mode visual intent and obsolete raw
  mask tokens.
- Repeated backward after an optimizer update to distinguish zero initialization
  from permanent disconnection.
- Audited gradients from the deployed action field separately from total loss.

【观察到的结果】

Initially only 845 of 1,169 trainable parameter groups had nonzero gradients.
Cleanup reduced the set to 894 and eliminated every `grad=None`; after one
normal update all remaining trainable parameters were nonzero. The only
action-root exclusions were the explicitly supervised proposal, JEPA, event and
motion output heads.

【当时的解释】

Dead compatibility branches were not harmless capacity: they made optimizer and
checkpoint identity misleading and allowed tests to certify the wrong graph.

【后来怎么看（截至冻结终点）】

Retained. Later schemas formalized module manifests and parameter-difference
reports for the same reason.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 1039-1056 and 1065-1068。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Optimizer owner, deployment root and manifest schema.

### CI-0133 — Proposal detach was a gradient-ownership bug with identical forward behavior

【时间 / 版本】

V103 action-history repair, 2026-07-26.

【当时在解决什么问题】

History proposal tokens participated in action forward but were detached, so
proposal queries and blocks could learn only from a weaker auxiliary objective.

【当时的 insight / hypothesis】

Preventing target leakage does not justify detaching a condition available at
deployment. A clean experiment can distinguish gradient ownership from function
value.

【具体做法】

Enabled natural action gradient to the proposal path under the V103 contract,
kept a legacy switch, and compared attached versus detached modes under the same
forward inputs. Added proposal-only keep-zero and shuffle probes while preserving
direct history memory.

【观察到的结果】

Forward outputs were bit-identical between modes; attached mode gave nonzero
final-action gradients to proposal queries/blocks, legacy mode gave
`grad=None`. Proposal-only zero and shuffle changed action and restored exactly
after clearing the intervention.

【当时的解释】

This was a pure backward-ownership correction, not evidence that proposal would
improve a trained policy.

【后来怎么看（截至冻结终点）】

Retained. It exemplifies why forward equivalence does not imply equivalent
learning dynamics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；确定性梯度所有权故障成立，不外推独立任务收益。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 1058-1076。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Action history, alias interventions and natural gradients.

### CI-0134 — Multi-slot existence required numerical diversity and consumer-use diagnostics

【时间 / 版本】

V103 probe completion, 2026-07-26.

【当时在解决什么问题】

Tests proved the configured number of address slots but could not tell whether
all slots represented the same posterior or whether policy used more than one.

【当时的 insight / hypothesis】

Multiplicity in shape is not multiplicity in function. Both producer separation
and consumer mass allocation need measurement.

【具体做法】

Added full-posterior Hellinger distance, center separation, effective slot count,
query-conditioned slot changes and G-to-W/W-to-P/P-to-MMDiT route-source counts.
Moved diagnostic capture before the deployment fast-return so probes observed
the actual sampled route.

【观察到的结果】

The checks could now distinguish copied slots and collapsed role routes without
changing forward, adding diversity loss or forcing quotas.

【当时的解释】

This converted a structural aspiration into a falsifiable post-training health
criterion.

【后来怎么看（截至冻结终点）】

Retained; later logs showed exactly why common-mode and interval-cosine metrics
were necessary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 1078-1089。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。

【关联】

Route common-mode, effective sources and typed innovation diagnostics.

### CI-0135 — Runtime contract validation caught a wrapper-chain error static tests missed

【时间 / 版本】

V103 launch, 2026-07-26.

【当时在解决什么问题】

The formal V103 launch aborted because an old V48 wrapper still enabled
`layer_recurrent_consequence`, contradicting the V103 graph.

【当时的 insight / hypothesis】

Testing a constructed config or the top wrapper fragment does not establish the
final CLI graph. Nested launchers must be expanded and validated at the actual
training entry.

【具体做法】

The V103 launcher explicitly disabled the inherited recurrent-consequence and
shared-FM probe paths. Wrapper-chain resolution was added to regression tests,
while the formal pre-training validator continued to reject all conflicts.

【观察到的结果】

The validator stopped before optimizer step one, so no invalid experiment was
produced. The assistant explicitly acknowledged that prior static tests had not
parsed the bottom wrapper's final values.

【当时的解释】

This was a successful fail-closed runtime defense around an incomplete static
audit, not proof that the audit had been complete.

【后来怎么看（截至冻结终点）】

Retained. Later manifests and resume rejection extended the same identity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 1090-1098。
- 段级工件：见 Segment 015 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`layer_recurrent_consequence`。

【关联】

Launcher inheritance, manifest identity and fail-closed startup.

## Segment 015 synthesis

V103 was the first large attempt in this history to rebuild the positive path
rather than merely close the latest shortcut. It retained spatial hypotheses,
made G/W/P residual ownership real, gave language/history exact null semantics,
introduced stateless phase queries and separated strict context masking from
future-delta targets. Yet the most valuable evidence came after the first
implementation “passed”: repeated reverse audits still found basis averaging,
far-anchor truncation, a hundredfold fixed-scale hierarchy, a spatially empty
flow floor, source raw that was never consumed, dead optimizer branches and a
detached deployment condition. A final wrapper-chain failure then showed that
even a correct model factory was not the launched graph. This segment therefore
strengthened a central lesson: completion requires the same identities and
semantics to survive source, training entry, loss, optimizer, deployment and
probe—not merely one successful forward/backward.

## Segment 016 — V103/V105 instability, V104 bounded recovery and interval-stage supervision

### Segment boundary

```text
Date:                 2026-07-26 to 2026-07-27
Visible messages:     1099-1300
Starts with:          early V103 training behavior
Ends with:            unified finite-Jacobian repair after V106 explosion
Primary historical artifacts:
  V103/V104/V105/V106 training logs
  scripts/current_v104_sequential_bounded_flow_jepa.sh
  sampling_path_probe.py
  V105 horizon soft-address path
  V106 interval-stage organizer and teacher targets
```

### CI-0136 — V103 exposed a coupled flow-escape, carrier-amplification and detail-drowning failure

【时间 / 版本】

V103 epoch one, 2026-07-26.

【当时在解决什么问题】

The newly reconstructed graph initially improved observable flow, but later
training showed growing world/P amplitudes, common future states and sudden
gradient spikes.

【当时的 insight / hypothesis】

Finite loss and bounded output updates do not imply safe internal geometry or
Jacobian. A model can move coordinates outside valid support, remove itself from
reconstruction evidence, expand a carrier and drown a protected residual.

【具体做法】

Tracked raw flow, valid sampling, warp gain, occlusion, world xy/anchor RMS,
protected-detail ratio and preclip gradients across the entire run rather than
the last batch.

【观察到的结果】

V103 had a healthy window near batches 200-800. Thereafter raw flow grew from
roughly `1.84` to hundreds, valid sampling approached zero, moving warp gain
became negative, world residuals reached thousands to tens of thousands, detail
ratio fell near `0.002`, and global norm rose through `10^9` to `10^17` before
overflowing at batch 2452. No validation epoch completed.

【当时的解释】

The graph was connected but admitted an escape route in coordinate/evidence
space. The resulting failure was structural and could not be cured by waiting
or reducing the logged loss.

【后来怎么看（截至冻结终点）】

Retained as the primary V104 stabilization target.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；confirmed training failure.

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 1099-1128 and 1145-1148。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。

【关联】

Bounded source-relative flow and residual-amplitude contracts.

### CI-0137 — Sequential horizon state must be an observable causal rollout, not parallel labels

【时间 / 版本】

V103 diagnosis and V104 design, 2026-07-26.

【当时在解决什么问题】

The `+4/+12/+24/+48` future states became increasingly similar despite different
time embeddings.

【当时的 insight / hypothesis】

`+48` should emerge from an ordered memory of perceived history and successive
near-to-far updates, not be a fourth parallel query initialized from the same
latest-motion seed with a different label.

【具体做法】

V104 read local observable motion history once and updated future states in the
order `history -> +4 -> +12 -> +24 -> +48` within one stateless forward. Future
teacher targets remained loss-only. The old V103 parallel path remained
reproducible behind its contract.

【观察到的结果】

Mechanism tests established causal order and teacher isolation. Early V104 logs
showed the sequential path was active and numerically stable, but later states
still became common after entering the broader world/trajectory carrier.

【当时的解释】

Sequential generation fixed one source of common-mode collapse but could not by
itself supply goal/phase-conditioned planning or guarantee downstream
differentiation.

【后来怎么看（截至冻结终点）】

Partly retained. Later review clarified that V104 constructed this memory before
language, action history and phase entered trunk, so it remained a visual future
context rather than a complete stage plan.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；数据探针；推断/未决
- 对话：visible messages 1108-1116, 1128-1144 and 1208-1212。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`+4/+12/+24/+48`。

【关联】

Interval stage targets and goal/phase-aware future organization.

### CI-0138 — V104 bounded values without hard clipping and recovered numerical viability

【时间 / 版本】

V104 implementation and run, 2026-07-26 to 2026-07-27.

【当时在解决什么问题】

V103 allowed source-relative flow to escape valid image geometry and allowed
typed/residual carriers to grow without an interpretable write budget.

【当时的 insight / hypothesis】

Stability should be enforced at the value/coordinate interface with smooth
contracts and native gradients, not by hard coordinate clipping, hard routing or
post-hoc gradient fabrication.

【具体做法】

- Mapped all flow levels into continuous bounded source-relative coordinates
  while preserving native grid units for sampling and normalized units for
  world motion keys/values.
- Applied smooth RMS write contracts to role sublayers, G/W/P bridges,
  P-to-bottom and protected detail.
- Combined this with the sequential future state as one explicit V104 identity;
  partial switch combinations were labeled ablations, not V104.

【观察到的结果】

At comparable training points, raw flow stayed near `1.4` rather than tens,
valid sampling remained `1.0`, moving warp gain was positive, world residuals
stayed single-digit and detail was not immediately drowned. V104 completed eight
epochs without NaN, OOM or the V103 collapse.

【当时的解释】

The repair successfully removed V103's catastrophic escape. It did not prove
long-horizon or conditional planning quality.

【后来怎么看（截至冻结终点）】

Retained as a clear stabilization success and later behavioral baseline.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 1128-1160 and 1204-1207。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。

【关联】

V106 finite-Jacobian contract; value bounds versus gradient bounds.

### CI-0139 — V104 improved to historical action range but kept several independent debts

【时间 / 版本】

V104 full eight-epoch log, reviewed 2026-07-27.

【当时在解决什么问题】

The user rejected an explanation centered only on normalized future and asked
for all log evidence to be revisited.

【当时的 insight / hypothesis】

One anomalous metric cannot explain long-horizon plateau, condition
homogenization, conservative gripper behavior and teacher-forced/deployment
gaps. Independent curves require independent causes unless evidence links them.

【具体做法】

Compared all epochs of training/action/representation/execution budgets,
validation bands, gripper events, goal/route diagnostics, sequential future
states and normalized/raw JEPA metrics.

【观察到的结果】

V104 broke an apparent epoch-two plateau at epoch three and approached stronger
historical action results. Yet improvement remained near/arm-heavy; tail-to-first
stayed large, gripper became conservative, goal-conditioned distinctions
weakened and sequential future states were built before goal/history/phase.
Training objectives kept improving after deployment validation flattened.

【当时的解释】

V104 was neither a failed nor a complete architecture. It recovered stability
and action competence while revealing missing conditional and stage ownership.

【后来怎么看（截至冻结终点）】

Retained. The user's insistence on looking beyond the most visible anomaly
prevented a one-cause redesign.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 1161-1212。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。

【关联】

Multi-metric audits, goal/phase conditions and gripper closure.

### CI-0140 — Raw and normalized future error answer different questions

【时间 / 版本】

V104 future-loss analysis, 2026-07-26 to 2026-07-27.

【当时在解决什么问题】

All raw `future_h*` errors improved while aggregate normalized `jepa_future`
worsened on validation.

【当时的 insight / hypothesis】

Raw error measures absolute reconstruction, while normalization by target-change
RMS magnifies relative error on weak changes. Opposite trends can reveal
redistribution toward large, easy changes rather than a logging bug.

【具体做法】

Traced the exact loss denominator, reliability and mask construction and compared
training versus validation, raw horizon errors and direction/scale objectives.

【观察到的结果】

Raw horizons and change direction improved while normalized validation future
rose. A context mask selected from current/history motion was reused across
future horizons, so the supervised current location could mismatch where a
future change moved; weak changes were also re-amplified when reliability was
normalized by its own total or by future-delta scale.

【当时的解释】

The fixed corruption mask was still necessary to prevent future-location leak.
The repair should add horizon-specific soft future address and loss-side
reliability, not reveal future target positions to the online input or learn a
zeroable binary mask.

【后来怎么看（截至冻结终点）】

Retained with the correction that context masking and future-address weighting
have different responsibilities.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 1178-1203。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`jepa_future`。

【关联】

V105 horizon soft address and reliability normalization.

### CI-0141 — V105's horizon address added supervision but recreated a small-variance Jacobian failure

【时间 / 版本】

V105 epoch one, 2026-07-27.

【当时在解决什么问题】

V105 reused the existing multiscale lattice to give each future horizon a soft,
observation-only spatial address without changing the deployed action graph.

【当时的 insight / hypothesis】

A unit-mass soft address avoids all-zero collapse and can remain broad when
uncertain, but broad averaging of signed high-pass residuals may create a
near-zero value whose subsequent scale-invariant normalization has an enormous
backward Jacobian.

【具体做法】

- Kept a fixed-ratio context mask and produced per-horizon softmax distributions
  over camera, source cells, slots and continuous multiscale candidates.
- Ensured teacher only supervised the posterior and never entered forward.
- Removed affine bias so a zero value bank produced exact zero write.
- Replaced self-normalized reliability with a smooth SNR relative to current
  teacher representation.
- Monitored horizon-address entropy/write and role gradients through epoch one.

【观察到的结果】

The new route remained very diffuse and weak early. From roughly batch 1840,
preclip gradients grew persistently; after batch 2200 grounding and upstream
visual/history gradients reached millions while loss and outputs stayed finite.
V104 with the same role hierarchy had remained stable, localizing the new trigger
to V105's horizon-address/small-variance path, amplified by repeated normalized
selectors.

【当时的解释】

This was a forward-bounded/backward-unbounded failure. It was not evidence that
soft addressing itself was invalid; variance floors had failed to cover the
full active Jacobian chain.

【后来怎么看（截至冻结终点）】

Retained as the immediate V106 numerical target.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；confirmed implementation failure.

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 1213-1220 and 1238-1248。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。

【关联】

Finite-gain routing, visibility continuity and interval-stage design.

### CI-0142 — Five-step deployment was self-consistent but off the linear training bridge

【时间 / 版本】

V104 frozen sampling-path probe, 2026-07-27.

【当时在解决什么问题】

The project needed to distinguish training/deployment call mismatch, recursive
state drift and numerical integration error as causes of poor far action.

【当时的 insight / hypothesis】

These are three different counterfactuals and can compensate. They must be
measured separately at matched time/noise rather than reduced to “5 versus 20
steps.”

【具体做法】

Compared teacher and deploy calls at the same exact bridge state, exact versus
recursive states at the same time, and 5/10/20-step integration under common
noise. The internal five-step implementation was required to equal public
`sample(5)` exactly.

【观察到的结果】

Call-contract excess MSE and teacher-versus-deploy prediction difference were
zero; internal and public five-step outputs were bit-identical. Recursive states
deviated from the linear bridge, but velocity error was often lower on the model's
own trajectory. The most abnormal behavior was the low-noise end of the exact
linear bridge, not simply coarse integration.

【当时的解释】

The model had learned a self-consistent discrete trajectory different from the
nominal linear training bridge. This ruled out several simple deployment-wiring
explanations without proving the learned trajectory was optimal.

【后来怎么看（截至冻结终点）】

Retained. The first probe run also failed because deployment fast-return omitted
diagnostics; that bug was fixed without changing model action.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 1221-1237。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。

【关联】

Flow-time geometry and later V120 sampler-contract restoration.

### CI-0143 — Interval stage targets replaced point anchors without averaging away motion

【时间 / 版本】

V106 design, 2026-07-27.

【当时在解决什么问题】

A single `+48` frame was a poor proxy for a phase, but plain multi-frame average
would erase motion direction and transient events.

【当时的 insight / hypothesis】

Future supervision should cover disjoint temporal intervals `4-8`, `8-16`,
`16-32`, `32-48` and combine robust interval content, signed temporal progression
and endpoint change while preserving camera and spatial cells.

【具体做法】

- Added a W-to-P interval organizer that read observable W evidence and organized
  only along the four temporal intervals.
- Kept duration encoding in query/key only; bias-free values came from W, so zero
  W evidence produced exact zero stage write.
- Supervised the same bounded delta written to the action path, eliminating an
  independent `progress_out` head that could learn the loss while write stayed
  zero.
- Kept interval state out of the high-resolution value bank and allowed it only
  to condition W and later queries.

【观察到的结果】

Two static review rounds found and fixed wrapper override, missing loss-ledger
integration, a fixed-template value shortcut and near-zero direction-normalization
singularities. A later first-batch crash exposed that teacher target construction
used an outer `self.anchors` field it did not own; preflight had tested deploy but
not the training target pack.

【当时的解释】

The interval formulation improved semantic ownership, but its implementation
again showed why a deployment-only smoke cannot certify a training-only teacher
path.

【后来怎么看（截至冻结终点）】

Retained structurally and later absorbed into four-interval Teacher/S/W
objectives. The original interval-target realization was not isolated as a
stability or performance cause.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；interval 语义保留，原 realization 未获独立归因。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 1221-1229 and 1248-1276。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`progress_out`、`self.anchors`。

【关联】

Teacher lifecycle, preflight coverage and later interval target algebra.

### CI-0144 — V106 delayed explosion proved value bounds did not bound the active Jacobian

【时间 / 版本】

V106 epoch-one partial run and repair, 2026-07-27.

【当时在解决什么问题】

V106 passed a 20-batch BF16/teacher smoke but developed catastrophic gradients
after a long stable prefix.

【当时的 insight / hypothesis】

The delayed onset revealed a state-dependent threshold crossing, not a stage
schedule or large interval loss. Residual RMS contracts bounded forward values
but not the Jacobian of normalization and hard visibility transitions upstream.

【具体做法】

- Located the first jump between batches 620 and 640 and aligned it with cycle,
  occlusion and correlation states.
- Audited active `F.normalize`, LayerNorm and visibility paths rather than every
  textual occurrence.
- Replaced active learned correlations with RMS floor `0.10` (max inverse gain
  10), role/address/memory normalization with variance floor `0.25`, bounded
  affine gain, and hard occlusion in backward with a continuous transition.
- Included block content modulation and final trunk normalization that the first
  pass had missed; kept hard visibility only as detached audit.

【观察到的结果】

At batch 600 global preclip was about `1.32`; at 640 it jumped to about
`3.74e3`, then reached `1e8-1e9`. The interval writer itself remained weak while
G/goal/history exploded first. The final static audit found no remaining naked
active normalizer of the same class under the V106-safe contract.

【当时的解释】

The error was deterministic enough to repair without another full experiment:
three interfaces—near-zero scale normalization, hard visibility flip and ordinary
role LayerNorm—formed a delayed high-gain chain.

【后来怎么看（截至冻结终点）】

Retained as a major numerical-contract lesson. A short smoke could not enter the
dangerous state, so stability validation must include adversarial low-variance
and threshold-boundary tests, not only early batches.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 1277-1300。
- 段级工件：见 Segment 016 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`F.normalize`。

【关联】

Gradient sentinel, first offending parameter and BF16/FP32 audits.

## Segment 016 synthesis

This segment separated three different notions of health. V104 showed that
bounded coordinates and smooth residual writes could rescue a catastrophically
unstable graph, yet its conditional and long-horizon semantics remained weak.
V105 showed that a conceptually reasonable soft future address could still
recreate a huge backward singularity through broad signed averaging and repeated
small-variance normalization. V106 improved temporal targets to interval-stage
increments, but a short smoke missed both a training-only target-construction bug
and a delayed Jacobian explosion. The durable lessons were that value bounds do
not bound gradients, teacher-only paths require their own preflight, and stage
supervision must share the exact online delta it intends to make useful rather
than train an independent head.

## Segment 017 — Stable V106, locally repaired V107 and progressive-grounding correction

### Segment boundary

```text
Date:                 2026-07-27
Visible messages:     1301-1420
Starts with:          V106 post-repair smoke and long-run validation
Ends with:            V109 progressive G1/G2/G3 implementation and static review
Primary historical artifacts:
  V106/V107/V108 attached training logs
  clearvla/policy/flow_dino_evidence.py
  clearvla/policy/trunk.py
  clearvla/policy/role_delta_attnres.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  scripts/current_v106_interval_stage_flow_jepa.sh
  scripts/current_v107_complete_top_path_flow_jepa.sh
  scripts/current_v109_progressive_grounding_address_flow_jepa.sh
```

### CI-0145 — Numerical safety exposed rather than solved functional non-use

【时间 / 版本】

V106 after the finite-gain repair, 2026-07-27.

【当时在解决什么问题】

The repaired V106 had to cross the old batch-600/620/640 danger region and show
whether its address, interval and detail paths recovered once the delayed
gradient explosion was removed.

【当时的 insight / hypothesis】

Bounding a forward value and its local Jacobian can make a graph numerically
safe while leaving the intended route functionally optional. A safety contract
becomes a bandwidth bottleneck when the network persistently requests a much
larger raw write, saturates the bound and can optimize through a stronger common
carrier instead.

【具体做法】

- Compared V104/V105/V106 at matched training batches instead of comparing an
  early trace with a final validation point.
- Separately tracked selector loss, selected value, actual write/carrier ratio,
  route entropy, common-mode cosine and per-group gradients.
- Followed V106 beyond the old explosion point and then through roughly batch
  1780 rather than relying on the 20-batch smoke.
- Kept the finite-gain repair and rejected restoration of unconditional
  LayerNorm merely to make a small value look large.

【观察到的结果】

- Global preclip remained roughly `1-2` across the old danger region; the V105/
  early-V106 catastrophic Jacobian did not recur.
- Learned flow became small but useful and late detail became selective.
- Horizon-address KL fell while its actual write was only about `0.001` at batch
  1780, versus about `0.039` for V105 at the same point.
- Interval write remained around `0.006` with a carrier ratio near `0.001`, and
  final interval representations remained almost parallel.
- Aggregated G/W/P raw residual demand rose to several times the actual write,
  while the written RMS stayed near the `0.5` ceiling.

【当时的解释】

V106 had fixed the numerical failure but not ownership. Extremely small
initialization, diffuse signed averaging, a small residual coefficient and a
complete common-carrier bypass jointly let the selector improve without making
the selected value necessary to action.

【后来怎么看（截至冻结终点）】

Retained. This is the point where “stable,” “trainable,” “informative” and
“actually consumed” became explicit independent review questions.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 1304-1323。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。

【关联】

V106 finite-gain repair, V107 post-gate contract and later functional-closure
reviews.

### CI-0146 — V107 repaired five local boundaries but did not make the routes indispensable

【时间 / 版本】

V107 design, implementation and early run, 2026-07-27.

【当时在解决什么问题】

The top-down source audit found five concrete online-path defects behind the
weak address and interval behavior.

【当时的 insight / hypothesis】

Local ownership errors should be repaired at their real write boundary before
changing losses: preserve multiple hypotheses and target-cell identity, give
interval innovation its own typed W-to-P source, constrain the post-gate write
rather than a pre-gate residual, and log each block rather than an aggregate.

【具体做法】

- Replaced the nominally multi-head/single-expectation reader with four real soft
  glimpses.
- Preserved the target 8x8 cell during fine horizon reads and chunked the read to
  control memory.
- Exported interval-stage innovation as a distinct typed W-to-P value.
- Moved the residual contract after the gate, so it bounded the tensor actually
  written.
- Added per-G/W/P raw, proposal and written diagnostics and corrected wrapper
  identity inheritance.

【观察到的结果】

Directed tests and action-only backward probes showed real gradients, exact-zero
semantics and active paths. The early V107 run nevertheless had extremely
diffuse glimpses, a rapidly growing world/anchor carrier and the weakest matched
late-detail write among the compared V101-V106 traces, while the main action
loss looked nearly unchanged.

【当时的解释】

V107 completed “the path exists” but only partly established “the path carries
unique information,” and did not establish “action must use it.” Local closure
and nonzero gradients were necessary but not sufficient evidence of system-level
closure.

【后来怎么看（截至冻结终点）】

Partly retained as a boundary repair. Its early result invalidated the stronger
claim that those five fixes alone completed the top path.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 1323-1362。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。

【关联】

Multi-glimpse P read, typed interval value, post-gate contracts and later
mandatory subsystem-familiarity rules.

### CI-0147 — “No second action bypass” was incorrectly turned into “do not write the existing bottleneck”

【时间 / 版本】

V105-V108 boundary correction, recognized during V107 analysis on 2026-07-27.

【当时在解决什么问题】

The user challenged the explanation that Future JEPA and horizon address were
merely weak auxiliary losses, noting that Future JEPA was supposed to be a
structural bottleneck.

【当时的 insight / hypothesis】

The safe requirement was to prohibit a second raw-detail-to-action decoder path,
not to prohibit an observation-owned address residual from writing the already
existing online rollout. Conflating those two statements converted a valid
anti-bypass rule into an information-discard rule.

【具体做法】

- Re-read the actual call order after the user's objection instead of defending
  the earlier metric interpretation.
- Distinguished the shared online rollout from the late
  `horizon_address_jepa` prediction-head residual.
- Traced the category error to the original configuration contract rather than
  treating it as a missing line introduced by training.
- Fixed four V108 tasks in writing: move the address read online before W,
  reuse the existing rollout, make Future JEPA consume that same final rollout,
  and retain an exact flags-off parent path.

【观察到的结果】

The source showed that the address residual had deliberately been prediction-
head-only and was skipped by deployment. Tests had validated that declared
contract, thereby certifying the wrong boundary rather than catching it.

【当时的解释】

The user characterized this as a low-level error and asked why it had not been
caught statically. That criticism changed the review requirement: a declared
contract could no longer be accepted as evidence that the contract itself was
semantically correct; the full consumer chain had to be explained.

【后来怎么看（截至冻结终点）】

Retained, including the interaction lesson. The frustration was triggered by a
specific, source-confirmed ownership error and led to a stricter review boundary;
the tone itself is not used as technical evidence.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 1361-1376。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`horizon_address_jepa`。

【关联】

Functional closure, anti-bypass semantics, V108 topology and later producer-to-
consumer/consumer-to-producer audits.

### CI-0148 — Moving the old reader online did not create progressive grounding

【时间 / 版本】

V108 implementation and early run, 2026-07-27.

【当时在解决什么问题】

V108 moved the horizon-address read from the terminal JEPA head to the G3-to-W1
online boundary and removed its duplicate terminal read.

【当时的 insight / hypothesis】

Correct call placement is not enough when the reader's internal abstraction was
designed as a weak auxiliary residual. Address formation must itself be organized
progressively; otherwise the same diffuse one-shot expectation is merely moved
earlier in the graph.

【具体做法】

- Used one online bank read after G3 and before W1.
- Reused the final rollout for Future JEPA and preserved the old flags-off path.
- Audited horizon cosine at the seed, G3, address and W3 boundaries, plus teacher
  entropy, source/slot/fine entropy, selected-value RMS and actual write.
- Revisited the intended G1/G2/G3 responsibilities after the early log contradicted
  the implementation claim.

【观察到的结果】

- Horizon cosine changed approximately `0.906 -> 0.981 -> 0.981 -> 0.995` across
  seed, G3, address and W3.
- The teacher address itself was broad (`~0.968` entropy), while the learned route
  remained broad and selected signed values cancelled to roughly `0.035 RMS`.
- The inherited auxiliary coefficient then produced an online write of only
  about `0.002` with a carrier ratio near `0.001`.
- G1/G2/G3 still processed a shared rollout; the actual address read happened
  once after all three and therefore could not support iterative hypothesis,
  rectification or rematerialization.

【当时的解释】

V108 fixed a deployment omission but not the target architecture. The main
failure was not the `0.10` coefficient alone: horizon identity had already been
diluted, the outer teacher did not supervise full fine correspondence, and
high-entropy signed aggregation erased value before the coefficient was applied.

【后来怎么看（截至冻结终点）】

Retained as a failed-mechanism baseline. It is a concrete example of a repair
that was locally correct yet only half-completed at the subsystem boundary.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已否定 as a complete grounding repair; its single-online-read correction was absorbed by V109.

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 1375-1403。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。

【关联】

Progressive grounding, high-entropy signed cancellation and late value read.

### CI-0149 — Progressive grounding requires hypothesis, rectification, rematerialization and canonical handoff

【时间 / 版本】

V109 design and implementation, 2026-07-27.

【当时在解决什么问题】

The G1/G2/G3 hierarchy needed real non-interchangeable responsibilities without
prematurely reading high-resolution values or allowing Flow to become the sole
address owner.

【当时的 insight / hypothesis】

Deterministic coordinate scaffolding belongs before G; G1 should establish
multiple correspondence hypotheses, G2 should rectify each hypothesis with
geometry and resample around the corrected center, and G3 should compile a
canonical selector state. Horizon-specific future posteriors belong to W, and
only P may aggregate the high-resolution value.

【具体做法】

- Pre-G retained camera identity, continuous coordinates and scale mappings.
- G1 formed per-camera, multi-slot coarse correspondences over the dense chart.
- G2 combined Flow, DINO/raw agreement, cycle, occlusion and uncertainty to move
  each hypothesis, then rematerialized N=49 fine candidates at the new center.
- G3 exported selector/key/coordinate/uncertainty state without value pooling.
- W formed horizon-specific priors and P remained the unique high-resolution
  reader.

【观察到的结果】

Static review found that merely changing G1/G2 logits would leave the old fine
candidates centered at the compiler location, so the supposed corrected address
could not reach corrected detail. It also found that a W temporal posterior had
initially been computed only for diagnostics after all blocks; it had to move to
the W3-to-P1 boundary and enter the same real P read.

【当时的解释】

An axis cannot be recovered by changing a score after the candidate set has
already discarded its support. Likewise, a posterior appearing in logs is not
functional unless it changes the unique value consumer.

【后来怎么看（截至冻结终点）】

Retained as the conceptual grounding boundary. V109 runtime did not prove the
whole realization healthy, but the same hypothesis→rectification→rematerialization
boundary later reappeared in the exact V120 source replay and Schema24 mechanical
restoration. The concept survived; no old V109 implementation is current authority.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；progressive-grounding concept retained, V109 realization superseded.

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 1403-1420。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。

【关联】

G1/G2/G3 ownership, candidate support, camera-axis preservation and unique P
value read.

### CI-0150 — Independent reverse review was what caught the half-connected paths

【时间 / 版本】

V107-V109 review process, 2026-07-27.

【当时在解决什么问题】

Several changes passed syntax, shape, zero-value and gradient tests while still
failing their intended system role.

【当时的 insight / hypothesis】

Review must run twice in opposite directions. Producer-to-consumer tracing finds
axis loss, bypasses and repeated reads; consumer-to-producer tracing finds
diagnostic-only state, dead optimizer parameters and gradients owned by the
wrong objective. Neither can be replaced by a named interface or a nonzero JVP.

【具体做法】

- Rechecked the complete forward chain after implementation rather than only the
  patched site.
- Independently started at the action/P consumer and traced every selected value,
  posterior, support and gradient to its producer.
- Removed old address/coarse-value/coarse-geometry modules that no longer
  executed but would otherwise remain optimizer-owned.
- Explicitly distinguished static/source review from unavailable CUDA runtime
  evidence.

【观察到的结果】

The reverse pass caught both major half-integrations in V109: G2 address changes
without candidate rematerialization and W temporal selection that existed only
after the real P consumer. It also prevented obsolete readers from surviving as
silent trainable branches.

【当时的解释】

The recurring failure mode was not primarily malformed tensors. It was a valid
local computation placed after its only meaningful consumer, or a new owner
whose support had already been destroyed upstream.

【后来怎么看（截至冻结终点）】

Retained and later generalized into the mandatory subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 1327-1336, 1381-1391 and 1410-1420。
- 段级工件：见 Segment 017 boundary 的 `Primary historical artifacts`。

【关联】

Bidirectional dataflow audit, optimizer ownership, diagnostics versus behavior
and post-edit closure.

## Segment 017 synthesis

V106 demonstrated that a numerically stable graph could still contain routes
that were optional, diffuse or saturated. V107 fixed five concrete local
boundaries but exposed the distinction between existence, unique information
and required consumption. The user's objection then uncovered a category error
embedded in the contract itself: preventing a second decoder bypass had been
misread as preventing address information from writing the existing online
bottleneck. V108 corrected that placement but only moved an auxiliary one-shot
reader, so its early log forced a second correction. V109 finally defined
progressive grounding as hypothesis formation, geometric rectification with
candidate rematerialization, canonical handoff, horizon-specific W selection and
one P-side high-resolution value read. The durable process lesson was that only
an independent forward and reverse audit exposed the half-connected states that
shape tests and nonzero gradients had repeatedly certified.

## Segment 018 — V109 long-run failure, V110 typed evidence and V111-V113 operational ownership

### Segment boundary

```text
Date:                 2026-07-27 to 2026-07-28
Visible messages:     1421-1703
Starts with:          V109 runtime evidence and the contract-as-agent-memory correction
Ends with:            V113 ownership/runtime review and delayed-instability hypotheses
Primary historical artifacts:
  V107/V108/V109/V110/V111/V112/V113 logs and launchers
  V109 progressive G/W/P source
  V110 native-RGB typed micro-grid reader
  V111-V113 public/private owner routing
```

### CI-0151 — The architecture contract is working memory, not an archive dump

【时间 / 版本】

V109 documentation correction, 2026-07-27.

【当时在解决什么问题】

The nominal compact contract had grown to roughly 1,748 lines and mixed the
active graph with version archaeology. It could preserve facts while still
being unusable as the agent's restart boundary.

【当时的 insight / hypothesis】

A source of truth must optimize for correct re-entry. The active execution
graph, tensor ownership, invariants, source entry points and unresolved
assumptions belong in a short front section; ancestry belongs behind an
explicit historical boundary.

【具体做法】

- Added an `Agent quick contract` at the top of the current contract.
- Separated active facts from the long historical audit.
- Updated workspace instructions to read the compact section first and old
  material only for ancestry questions.
- Rejected a new versioned contract for every architecture turn.

【观察到的结果】

The current V109 graph could be recovered from a bounded section without
reconstructing it from experiment names or hundreds of old decisions.

【当时的解释】

Information completeness and retrieval fitness are different properties. A
document can contain every fact and still encourage the wrong reconstruction.

【后来怎么看（截至冻结终点）】

Retained as the ancestry of the current compact-contract and auxiliary-ledger
separation.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录；历史检索
- 对话：visible messages 1422-1425。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Repository memory, version identity and post-compaction recovery.

### CI-0152 — Exact current geometry and predicted future geometry have different truth conditions

【时间 / 版本】

V109-to-V110 spatial design, 2026-07-28.

【当时在解决什么问题】

V109 retained an `84x84` learned raw feature map, but not literal RGB detail,
and treated current observation addresses and future locations too similarly.

【当时的 insight / hypothesis】

Current RGB is observed and can be bound to exact camera, coordinate, scale and
validity. Future RGB does not exist online; its legal geometry is a distribution
attached to an observed anchor: displacement, scale change, visibility and
covariance. Pooling the current image destroys available truth, while demanding
an exact future pixel invents unavailable truth.

【具体做法】

- Kept low-resolution DINO as the global semantic coordinate system.
- Proposed current native RGB/detail as a coordinate-addressable patch bank.
- Used G/W to predict a soft future transport from current anchors rather than
  a fictitious future pixel identity.
- Preserved the bottom interface by making P the query-conditioned local
  consumer.

【观察到的结果】

The design rejected both extremes: a positionless feature bag and putting the
entire DINO encoder after an immature patch selector.

【当时的解释】

The representation boundary must follow what is observable at each lifecycle
point. The same tensor shape does not make current and future coordinates the
same semantic object.

【后来怎么看（截至冻结终点）】

Implemented in V110 as exact current local reads plus predicted future
transport. Later logs showed that merely preserving the geometry field did not
guarantee action use; camera/transport ownership was repeatedly repaired through
V120 and Schema24-39. The truth-condition distinction remained valid while the
V110 realization was not a sufficient closure.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；geometry truth contract retained, V110 action benefit unproved.

【证据与锚点】

- 类型：日志观测；方案/设计记录；推断/未决
- 对话：visible messages 1440-1449。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

P geometry, Teacher transport, camera identity and literal-detail ownership.

### CI-0153 — Early overfitting was supporting evidence for detail non-use, not proof by itself

【时间 / 版本】

V104-V110 long-run comparison, 2026-07-28.

【当时在解决什么问题】

Several runs reached their best validation point early and then regressed. The
user proposed that failure to use fine detail was the cause.

【当时的 insight / hypothesis】

An early validation rebound alone is non-identifying. It becomes strong evidence
of detail non-use only when paired with matched address interventions, falling
late-detail share, diffuse W/P posteriors and the horizon/channel pattern of the
regression.

【具体做法】

- Compared full validation trajectories rather than the best epoch only.
- Reused V98 zero/shuffle evidence showing tiny action change from address
  interventions.
- Tracked V109 late-detail ratio falling from about `0.023` through `0.013` to
  roughly `0.007-0.010`.
- Separated the upstream exploding Flow/G graph from the still-weak final
  detail reader.

【观察到的结果】

Coarse action mappings learned quickly, while tail, gripper and event behavior
regressed and the precise route did not take over later optimization.

【当时的解释】

The model could finish the easy low-frequency solution and continue fitting it
without acquiring the local evidence needed for better generalization.

【后来怎么看（截至冻结终点）】

Retained as a multi-evidence diagnosis, not a rule that every late rebound is a
detail failure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留 with the stated evidential limit.

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；方案/设计记录
- 对话：visible messages 1450-1451。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Generalization, matched interventions and fine-action bandwidth.

### CI-0154 — Preserving typed tensors is weaker than preserving typed decisions

【时间 / 版本】

V110 review and V111 design, 2026-07-28.

【当时在解决什么问题】

V110 carried semantic, appearance and geometry projections, yet the resulting
world and policy behavior remained strongly public and nearly insensitive to
typed conditions.

【当时的 insight / hypothesis】

Types can survive as named tensors while losing functional ownership if their
logits are repeatedly summed into one posterior, their values are pooled by the
same weights or their conditions are added into one token before the real
consumer.

【具体做法】

- Traced G2, G3, W, P1 and P2 through every typed merge.
- Found separate projections feeding shared posteriors and shared slot weights.
- V111 introduced owner-specific posteriors, sidecars, two-stage P1 selection,
  distinct RGB/detail values and owner-specific interventions.

【观察到的结果】

V110 had real raw/flow/detail learning but P2 typed-condition gradient was about
`1.8e-5` versus a P2 refiner gradient around `4e-2`; typed identity existed but
decision authority did not. V111 improved local owner gradients but did not yet
make W/P use those owners indispensably.

【当时的解释】

Information preservation must include the probability measure and terminal
consumer, not only the value tensor or module name.

【后来怎么看（截至冻结终点）】

Retained. Later versions repeatedly revisit the same distinction as sidecar
versus operational owner.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预
- 对话：visible messages 1500-1529 and 1530-1559。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

S/P selection quality, common publicization and typed posterior ownership.

### CI-0155 — A structured bottleneck restricts access, not information capacity

【时间 / 版本】

V110-to-V111 design discussion, 2026-07-28.

【当时在解决什么问题】

The public carrier dominated useful bandwidth, but replacing it with a single
small token or averaged private vector would have removed the very geometry and
detail the repair was meant to protect.

【当时的 insight / hypothesis】

A useful private bottleneck limits who may read or write and where evidence can
enter the action graph. It need not collapse candidate, coordinate, type or
interval capacity. A crude low-rank bottleneck turns ownership into global
information loss.

【具体做法】

- Defined public state as stable scene/task/common-horizon information.
- Defined private evidence as a structured bundle of multiple addresses,
  local values, transport, confidence and interval innovation.
- Kept final action-query selection at P rather than pre-averaging the bundle.

【观察到的结果】

This reframed the design target from increasing weak-branch gain to conserving
private axes until a named terminal consumer.

【当时的解释】

Ownership is an access-control topology. Capacity destruction is neither
necessary nor sufficient to establish it.

【后来怎么看（截至冻结终点）】

Retained as a central information-conservation principle.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 1507-1517。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Global bottlenecks, protected bases, optional innovation and K/camera axes.

### CI-0156 — Memory optimization should factor computation, not delete evidence

【时间 / 版本】

V110 native-resolution P reader, 2026-07-28.

【当时在解决什么问题】

The first V110 micro-grid implementation caused a large training-memory
increase. The temptation was to reduce candidates, resolution or ownership
lanes.

【当时的 insight / hypothesis】

Activation topology, not the existence of high-resolution evidence, was the
immediate problem. Materializing query x glimpse x camera x cell x slot x
candidate x micro-cell state for every chunk retained redundant backward
activations.

【具体做法】

- Counted the actual candidate and micro-grid axes.
- Reordered the algebra to aggregate sufficient statistics before micro-cell
  expansion.
- Used query chunking and activation recomputation.
- Kept all typed lanes, N=49 candidates and the `3x3` micro-grid.

【观察到的结果】

The micro-reader CUDA peak increment fell from about `380.1 MiB` to `91.2 MiB`
per measured chunk, while output max error stayed about `8.94e-8` and gradient
max error about `1.19e-7`.

【当时的解释】

Compute savings are valid when they exploit distributivity and lifetime, not
when they silently redefine the evidence support.

【后来怎么看（截至冻结终点）】

Retained as the preferred optimization order; the overall full-model cost still
required production measurement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 1474-1494 and 1579-1587。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Activation lifetime, query chunking and information-preserving efficiency.

### CI-0157 — Repairing a sidecar can expose a worse intermediate policy

【时间 / 版本】

V111-V112 comparison, 2026-07-28.

【当时在解决什么问题】

V112 greatly increased appearance gradients and reduced public ratio, yet its
epoch-one action validation was modestly worse than V111.

【当时的 insight / hypothesis】

A semantic repair can weaken an old compensating shortcut before the intended
replacement becomes a complete decision path. This produces a legitimate
intermediate regression: representation health improves while action utility
falls.

【具体做法】

- Made G3 public chart independent of averaged private owners.
- Preserved semantic/appearance/geometry/interval sidecars across W.
- Made W appearance affect P1 fine candidates.
- Compared V111 and V112 only on the one complete common epoch.

【观察到的结果】

W appearance gradient rose from roughly `5e-6` to `1.6e-3` and public ratio
fell from about `0.862` to `0.796`, but validation action RMSE worsened about
`2.5%` and gripper F1 about `7%`. The typed path was connected but still entered
the action graph as an optional correction.

【当时的解释】

Local semantic correctness is not the same as system closure. The correct
question is whether the repaired owner reaches the terminal operation that the
old shortcut had performed.

【后来怎么看（截至冻结终点）】

V113 attempted to turn sidecars into operational boundary routing rather than
reverting the ownership repair.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V112 is an informative intermediate, not a performance anchor.

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 1564-1578 and 1628-1648。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Positive-path closure, shortcut removal and staged structural migration.

### CI-0158 — Supervision after the consumer creates a parallel success path

【时间 / 版本】

V113 implementation review, 2026-07-28.

【当时在解决什么问题】

Per-horizon phase/goal/history and W owner routes entered the online W blocks,
but the interval objective was still produced by a separate organizer after W3.

【当时的 insight / hypothesis】

An auxiliary predictor can satisfy the intended supervision while the online
state remains unnecessary. The supervised field must be the same field consumed
by P, not a post-hoc sibling with easier access to the target semantics.

【具体做法】

- Replaced the post-W organizer target with direct supervision of the routed W
  interval candidates.
- Made W3-to-P ownership writes explicit candidates rather than an implicit
  perturbation of public rollout.
- Removed the direct policy-to-appearance scorer that bypassed the W verifier.

【观察到的结果】

V113 improved P1/P2 ownership locally, but early logs still showed W interval
gradient about `30-50x` below semantic/appearance/geometry and nearly uniform W
functional routing.

【当时的解释】

Correct loss placement restores the possibility of ownership; it does not by
itself guarantee identifiable interval variation on simple data.

【后来怎么看（截至冻结终点）】

Retained as the distinction between objective success and online-field success.
V113 and many later schemas repeatedly confirmed it: moving the loss onto the
online field was necessary, but W could still remain weak, public or optional at
the final action consumer.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V113 realization remained functionally weak.

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 1594-1610 and 1642-1655。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Teacher fences, S/W interval ownership and diagnostic-only predictions.

### CI-0159 — Delayed gradient explosion was a learned high-gain state transition, not useful detail itself

【时间 / 版本】

V108-V110 retrospective during V113, 2026-07-28.

【当时在解决什么问题】

Several versions trained normally for long prefixes and then developed
catastrophic top gradients. The user asked whether the transition represented a
late attempt to learn fine detail.

【当时的 insight / hypothesis】

The weaker claim was plausible: once the coarse solution saturated, optimization
could press on the fine route. The strong claim was rejected. The attained state
was a high-gain, low-stability strategy that exploited raw scale, normalization
or routing sensitivity rather than a successful fine representation.

【具体做法】

- Aligned the transition sequences of V108, V109 and V110.
- Separated final bounded write from raw internal state and Jacobian.
- Traced which gradient groups moved first and which remained small.
- Compared detail entropy/write with warp gain, occlusion, cycle and action
  interventions.

【观察到的结果】

V108 moved from global norm about `1.9` to `62`, `1.57e5` and `2.75e10` while
write RMS remained near `0.45`. V109 showed DiT/W/Flow/G gradients rising before
raw flow and occlusion deteriorated. P2, W-to-P and bottom policy did not grow in
proportion. Brief detail sharpening preceded failure, but moving warp and
geometry quality then collapsed.

【当时的解释】

The model may have discovered that increasing an internal proposal could gain
routing control even when the forward write was contracted. That is genuinely
learned behavior, but it exploits a numerical degree of freedom instead of
delivering useful evidence.

【后来怎么看（截至冻结终点）】

Retained. Correct redirection means making address differentiation and action
utility—not raw magnitude—the available order parameter.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；exact trigger batch replay was lost with the deleted checkpoint.

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 1634-1689。
- 段级工件：见 Segment 018 boundary 的 `Primary historical artifacts`。

【关联】

Finite Jacobians, scale gauges, alert checkpoints and S/P functional closure.

## Segment 018 synthesis

V109-V113 clarified two failure families that had repeatedly been conflated.
Fine evidence could remain unconsumed even in a numerically stable run, while a
separate upstream high-gain state transition could explode without strengthening
the final P/action route. V110 preserved literal local structure but repeatedly
merged typed decisions; V111/V112 preserved private sidecars but left them
optional; V113 moved supervision and routing closer to the real consumer but
still began with weak interval ownership. The durable design rule is therefore
threefold: preserve observable axes, supervise the exact online field and prove
its terminal action use. Bounded forward RMS, named typed tensors and nonzero
gradients each cover only one part of that obligation.

## Segment 019 — V113-V116 causal localization, future-effect semantics and system boundaries

### Segment boundary

```text
Date:                 2026-07-29
Visible messages:     1704-2054
Starts with:          V113 delayed-explosion follow-up and matched causal probes
Ends with:            V116 implementation/static-review completion and experimental handoff
Primary historical artifacts:
  V112 checkpoints and updated V113/V114/V115 logs
  clearvla/policy/trunk.py
  clearvla/policy/flow_dino_evidence.py
  clearvla/policy/system.py
  clearvla/policy/time_domain_mmdit.py
  V115/V116 contracts, probes, launchers and tests
```

### CI-0160 — A conspicuous numerical failure must not consume the whole architecture review

【时间 / 版本】

V112-V113 retrospective, 2026-07-29.

【当时在解决什么问题】

V112 developed a delayed gradient explosion, while V111 remained stable and
V113 appeared healthier through its first several epochs. The immediate danger
was to let the explosion become the only question asked of the run.

【当时的 insight / hypothesis】

Numerical stability, representation usefulness and downstream consumption are
independent audit axes. A run can be finite while its intended W/P route is
harmful, and it can explode for a learned high-gain reason while another part of
the graph remains genuinely useful.

【具体做法】

- Kept V111 as a numerical control.
- Examined V113 gradients and action validation separately from matched causal
  interventions.
- Required the review to cover G, W, P1/P2 and temporal organization even after
  locating the most visible spike.

【观察到的结果】

V113 was numerically healthy through the inspected prefix, but later matched
interventions showed that deleting W-side contributions improved action MSE.
The absence of an explosion therefore did not establish a healthy architecture.

【当时的解释】

The strongest failure signal and the most important causal defect need not be
the same phenomenon.

【后来怎么看（截至冻结终点）】

Retained as a review-method rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预
- 对话：visible messages 1704-1745。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Numerical-state transitions, matched interventions and whole-graph review.

### CI-0161 — Matched interventions localized useful G, harmful W delivery and temporal load concentrated in P2

【时间 / 版本】

V113 frozen-checkpoint causal probing, 2026-07-29.

【当时在解决什么问题】

Aggregate gradients and internal RMS values could not answer which top modules
actually improved the deployed action.

【当时的 insight / hypothesis】

Only interventions performed at matched semantic boundaries can distinguish a
useful owner from a tensor that is merely active. The action error, not an
internal proxy, must be measured after each intervention.

【具体做法】

- Zeroed the G3 address contribution at its real consumer.
- Removed W shared residual and W-to-P delivery at their realized boundaries.
- Removed the P1 delta separately.
- Shuffled P2 temporal organization and policy temporal organization without
  changing tensor shape or total scale.

【观察到的结果】

- `address_g3_zero` worsened action MSE by about `10.03%`.
- Removing W shared residual improved MSE by about `10.65%`.
- Removing W-to-P and P1 delta improved MSE by about `2.17%` and `1.96%`.
- P2 temporal shuffle worsened MSE by about `52.99%`.
- Policy temporal shuffle worsened MSE by about `74.13%`.

【当时的解释】

G3 address information was useful; the realized W delivery was actively
harmful; and the surviving temporal organization burden had been displaced into
P2/policy rather than being carried by W as intended.

【后来怎么看（截至冻结终点）】

Retained as one of the first strong functional localizations of the G-W-P graph.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 1715-1764。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`address_g3_zero`。

【关联】

CI-0158, action-side causal probes and functional ownership.

### CI-0162 — Supervising a prediction head does not supervise the hidden state consumed by policy

【时间 / 版本】

V113 source audit, 2026-07-29.

【当时在解决什么问题】

Future auxiliary loss looked trainable, yet the W contribution consumed by P
was harmful and weakly identified.

【当时的 insight / hypothesis】

A projection can learn its target while the sibling hidden carrier used by the
policy retains unrelated, shortcut or action-echo information. “The future head
learned” and “the policy consumes the supervised future state” are different
contracts.

【具体做法】

- Traced W inputs and found access to noisy action/proposal information.
- Traced the future loss to a W prediction projection.
- Independently traced P to an incompletely constrained W hidden carrier.
- Compared the target-bearing field and policy-bearing field rather than
  assuming module-name identity implied tensor identity.

【观察到的结果】

W could echo action/proposal state into its policy carrier while satisfying the
future head through a different projection. This explained why removing W could
improve action despite a decreasing future objective.

【当时的解释】

Loss ownership must terminate on the same typed field that the downstream
consumer reads, or the loss creates a parallel proof of success.

【后来怎么看（截至冻结终点）】

Retained; it directly motivated the later `FutureEffectField` boundary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 1750-1786。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

CI-0158, future-state ownership and shortcut carriers.

### CI-0163 — Block allocation should follow distinct causal work, not inherited symmetry

【时间 / 版本】

V113-to-V115 planning, 2026-07-29.

【当时在解决什么问题】

The inherited top hierarchy used a `3-3-2` G-W-P block allocation even though
causal probes showed unequal functional load.

【当时的 insight / hypothesis】

Depth is justified by distinct transformations and supervised consumption, not
by visual symmetry between named modules. G retained three progressive spatial
jobs; W's third block had not demonstrated a unique role and became more public
with depth; P had only two blocks while performing fact reading, typed
operations, temporal organization and action correction.

【具体做法】

The working architecture hypothesis shifted from `G3-W3-P2` to `G3-W2-P3`,
subject to preserving the useful operations rather than simply moving parameter
counts.

【观察到的结果】

This reframed the redesign around causal burden. It did not claim that three or
two blocks were universally optimal; it claimed that the old allocation lacked
functional evidence.

【当时的解释】

A block that only produces a more public version of an already available state
is not made necessary by calling it a hierarchy stage.

【后来怎么看（截至冻结终点）】

Partially retained as historical rationale; later mainlines changed the exact
module semantics again.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；具体深度不是当前授权。

【证据与锚点】

- 类型：历史检索；方案/设计记录；推断/未决
- 对话：visible messages 1765-1794。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Functional load, owner depth and V115/V116 ancestry.

### CI-0164 — W was redefined as a goal-conditioned successor effect on current-G coordinates

【时间 / 版本】

V114-V116 design, 2026-07-29.

【当时在解决什么问题】

Predicting a fixed future-frame DINO representation made identity, camera and
time correspondence ambiguous and encouraged a public average.

【当时的 insight / hypothesis】

W should predict what may change relative to the currently grounded objects,
not reconstruct an arbitrary future image latent. The intended object includes
semantic change, spatial transport, persistence/visibility and uncertainty,
conditioned on the goal/phase while remaining a factual future hypothesis.

【具体做法】

- Used current G as the reference support.
- Associated multiple future supports to current objects.
- Defined successor content and semantic delta relative to current reference.
- Represented transport and status-like changes separately.
- Kept Goal/Phase responsible for desired/active task state and W responsible
  for possible physical consequences.

【观察到的结果】

The resulting `FutureEffectField` made a source-level supervised-consumer
contract possible, although later logs showed that its first realization still
contained bypasses and degenerate teacher reliability.

【当时的解释】

A future latent becomes identifiable when its reference frame, identity support
and downstream operation are explicit.

【后来怎么看（截至冻结终点）】

Retained as an important semantic correction, while later revisions changed the
teacher algebra and exact status treatment.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；历史 field details are not current authority.

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 1778-1831 and 1980-1986。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Teacher association, successor effects and G-aligned coordinates.

### CI-0165 — Current G can organize future supervision without becoming the future predictor

【时间 / 版本】

V114 future-teacher design, 2026-07-29.

【当时在解决什么问题】

Future observations had no stable object identity or spatial coordinate shared
with the online policy.

【当时的 insight / hypothesis】

The completed current G state can serve as the teacher's object/address chart.
Future evidence is associated back to this chart under no-grad, while online W
predicts the resulting effect. This avoids giving deployment access to future
frames and avoids inventing a second object coordinate system.

【具体做法】

- Built teacher targets only after completed G3.
- Kept future encoding and association in the training-only teacher.
- Exported targets aligned to current object slots and spatial support.
- Required deployment teacher calls to remain zero.

【观察到的结果】

The lifecycle was conceptually sound, but an implementation later materialized
a `B x support x camera x current-cell x slot x future-cell x hidden` FP32
dispersion tensor and exhausted memory.

【当时的解释】

Coordinate ownership and computational evaluation order are separate
requirements: the correct teacher can still be implemented with a catastrophic
intermediate.

【后来怎么看（截至冻结终点）】

Retained together with the requirement to compute moments through sufficient
statistics rather than dense pairwise hidden differences.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 1790-1838 and 1939-1952。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Teacher isolation, current-reference coordinates and CI-0156.

### CI-0166 — Stateless phase means re-inferred belief, not absence of sequential evidence

【时间 / 版本】

V114-V115 Goal-Phase design, 2026-07-29.

【当时在解决什么问题】

A single mean-pooled goal token could not express task progress or distinguish
long-horizon intervals, but maintaining a recurrent hidden state would complicate
deployment and identity.

【当时的 insight / hypothesis】

A controller can be stateless across calls while performing sequential
inference inside each call. It can reconstruct a phase belief from the full T5
sequence, observed state history, executed-action history, flow/history evidence
and current grounding.

【具体做法】

The design moved toward a Goal-Phase Belief Machine with active, next and
remaining goal evidence rather than adding one pooled vector to every W query.

【观察到的结果】

V115 logs later showed a different failure: the phase path quickly collapsed to
terminal probability near one and transition gradients approached zero. This
did not invalidate stateless belief; it showed that the realized supervision and
consumer topology still admitted an easier terminal shortcut.

【当时的解释】

Statelessness is a lifecycle property, not an information-capacity restriction.

【后来怎么看（截至冻结终点）】

Retained as a conceptual distinction; the particular V115 phase implementation
was not accepted as healthy.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留概念，具体实现被后续修订。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 1815-1848 and 1986-2017。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

History conditioning, phase collapse and recurrent versus recomputed state.

### CI-0167 — An intervention that broadcasts successfully can still violate the axis contract

【时间 / 版本】

V113/V114 probe repair, 2026-07-29.

【当时在解决什么问题】

A uniform-route intervention ran through ordinary broadcasting but failed in
the typed micro-grid reader.

【当时的 insight / hypothesis】

Broadcast compatibility is not shape correctness. A probe must carry every
real query, glimpse, camera, slot and candidate axis present at the boundary it
claims to perturb.

【具体做法】

- Found that the uniform posterior omitted explicit query/glimpse axes.
- Replaced implicit broadcasting with `expand_as(baseline_route_weights)`.
- Added explicit intervention mode, batch and shape reporting.
- Required the diagnostic path itself to pass the same typed-axis tests as the
  production path.

【观察到的结果】

The corrected probe preserved the realized posterior shape and no longer relied
on a semantically false broadcast.

【当时的解释】

Diagnostics are part of the measurement system. A malformed intervention can
produce confident but meaningless causal conclusions.

【后来怎么看（截至冻结终点）】

Retained as a probe-design invariant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 1840-1870。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Typed axes, matched interventions and shape provenance.

### CI-0168 — Fact queries can share expensive reading work without sharing away action-basis decisions

【时间 / 版本】

V114 P1 efficiency experiment, 2026-07-29.

【当时在解决什么问题】

P1 used `96 = 24 horizons x 4 action bases` high-resolution factual queries,
making the micro-grid read expensive.

【当时的 insight / hypothesis】

The expensive visual fact read need not be repeated for every action basis if
basis identity is reintroduced at the later decision consumer. However, sharing
the factual query does not authorize averaging away spatial or typed variation
in the result.

【具体做法】

- Reduced P1 to 24 factual queries.
- Preserved N=49 posterior candidates, four glimpse types and the `3x3`
  RGB/detail micro-grid.
- Deferred basis-specific organization to P2.
- Kept the existing chunking/checkpoint strategy.

【观察到的结果】

Training throughput improved from roughly `4.4 s/batch` in V113 to about
`2.08 s/batch`. But late-detail RMS fell from about `0.054` to `0.022`, glimpse
variation approached zero and P2 owner delta fell from about `0.034` to zero.

【当时的解释】

The compute factorization was valuable, but the first shared read publicized the
fact state too early. Efficient sufficient statistics and retained decision
identity had to be designed together.

【后来怎么看（截至冻结终点）】

Retained as a mixed result: query de-duplication was useful; result
publicization was not.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留，V114 realization superseded.

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 1830-1882 and 1931-1938。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`4.4 s/batch`、`2.08 s/batch`。

【关联】

CI-0156, P1 factual reading and delayed action-basis conditioning.

### CI-0169 — Tensor algebra must be audited for peak intermediates, not only output shape and FLOPs

【时间 / 版本】

V115 first production start, 2026-07-29.

【当时在解决什么问题】

Preflight passed, but batch-eight training failed while computing Teacher-G
dispersion.

【当时的 insight / hypothesis】

An algebraically simple broadcast can dominate memory even under `no_grad`.
Output-size accounting and isolated preflight do not reveal the peak when the
full action backward graph is already resident.

【具体做法】

- Expanded the actual broadcast axes and calculated the intermediate size.
- Identified a tensor of shape conceptually
  `[B=8,S=12,C=2,8,8,K=4,8,8,H=512]` in FP32.
- Replaced pairwise hidden differences with an equivalent second-moment
  identity.
- Preserved all supports, slots, cells and target semantics.
- Added numerical-equivalence and production-shape tests.

【观察到的结果】

The single intermediate required exactly about `6 GiB`. The second-moment form
removed it without reducing evidence; V115 targeted tests and lightweight
regressions passed.

【当时的解释】

Memory safety is a property of evaluation order and activation lifetime, not
only model capacity. A preflight must reproduce the real lifecycle or explicitly
state what it omits.

【后来怎么看（截至冻结终点）】

Retained as a canonical example of information-preserving factorization.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 1939-1952。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`no_grad`。

【关联】

CI-0156, Teacher lifecycle and production-memory tests.

### CI-0170 — Global conditions need typed expansion helpers rather than repeated ad-hoc broadcasting

【时间 / 版本】

V115 P3 startup repair, 2026-07-29.

【当时在解决什么问题】

A `[B,H]` phase/goal condition was expanded directly against a
`[B,24,4,H]` basis tensor with a missing time axis, causing a runtime mismatch
between sizes 8 and 24.

【当时的 insight / hypothesis】

Repeated manual `None` insertion is not a stable interface. A global condition
must cross one explicit `[B,H] -> [B,T,K,H]` boundary that owns time and basis
semantics.

【具体做法】

- Added a single expansion interface for active, next and remaining goal plus
  phase uncertainty.
- Covered the production `B8/T24/K4` shape in tests.
- Reviewed dtype and zero semantics at the same boundary.

【观察到的结果】

The startup error was removed, but its occurrence alongside the 6 GiB teacher
intermediate showed that local tensor reasoning had become difficult in the
monolithic implementation.

【当时的解释】

Named dataclasses do not protect axes unless the transformations crossing those
interfaces are themselves centralized and tested.

【后来怎么看（截至冻结终点）】

Retained; this fed directly into the later mandatory subsystem-familiarity and
typed-boundary discipline.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 1943-1955。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`B8/T24/K4`。

【关联】

Axis identity, P3 conditions and implementation complexity.

### CI-0171 — Architecture complexity and research-system complexity are distinct debts

【时间 / 版本】

V115 refactor discussion, 2026-07-29.

【当时在解决什么问题】

Shape, dtype, memory, probe, launcher, serialization and checkpoint errors were
appearing across multiple files even when the intended network idea was
coherent.

【当时的 insight / hypothesis】

The project had become a monolithic research system, not merely an overcomplicated
network. Model graph, experiment inheritance, CLI/configuration, training and
deployment lifecycle, Teacher/cache precision, losses, diagnostics and legacy
compatibility were entangled.

【具体做法】

- Distinguished stable primitives from architecture-specific composition.
- Proposed thin shared experiment/training/checkpoint protocols rather than a
  universal model with historical feature flags.
- Proposed isolating only meaningful architecture baselines and treating the
  current ClearVLA as one explicit graph.
- Identified V98, V107 and the then-current V115 line as potentially meaningful
  architectural anchors rather than preserving every intermediate patch version.

【观察到的结果】

The discussion established a refactor boundary, but experiment evidence then
showed V115 was not healthy enough to freeze as the new foundation. Structural
repair was prioritized before large-scale extraction.

【当时的解释】

Code modularity should make local reasoning possible; it must not fossilize an
unverified graph or reproduce it three times behind different directory names.

【后来怎么看（截至冻结终点）】

Retained as systems guidance. The exact proposed directory layout was an early
plan, not current authority.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留原则；具体重构布局尚非当前契约。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录；方案/设计记录
- 对话：visible messages 1953-1971。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Experiment identity, architecture registries and lightweight contracts.

### CI-0172 — Joint metric chronology can prove a shortcut before a bespoke probe does

【时间 / 版本】

V115 early-log reinterpretation, 2026-07-29.

【当时在解决什么问题】

Early analysis overemphasized a proposed fixed-time action probe even though the
existing log already contained a strong temporal ordering of failures. The user
explicitly challenged the review for not reading the joint evidence carefully
enough.

【当时的 insight / hypothesis】

When several diagnostics change in a consistent order, chronology itself can
exclude causal stories. A later Teacher collapse cannot explain an earlier P3
shortcut saturation.

【具体做法】

The log was re-read as a single timeline:

- FutureEffect head had gradient while W blocks were near zero by batch 20-200.
- P3 terminal, temporal, effect and precision lanes approached their ceilings by
  roughly batches 120, 240, 300 and 480.
- G3 semantic/appearance ownership collapsed only around batch 780.
- Teacher reliability fell below about `0.01` around batch 820.
- Phase became terminal and transition gradient vanished around batch 940.
- Action loss then alternated widely through batch 1200 and later.

【观察到的结果】

The ordering showed that P3 had learned an early bypass before Teacher-G and
phase fully degraded. The FutureEffect summary, P2 delta, goal and basis were
mixed in a lane labelled as effect, so the lane did not belong uniquely to W.

【当时的解释】

Additional probes can refine attribution, but they should not postpone a
conclusion already supported by source topology plus synchronized diagnostics.

【后来怎么看（截至冻结终点）】

Retained both technically and as a review-process correction. The user's loss of
confidence changed the requirement from single-metric interpretation to
source-backed, chronological joint analysis.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 1972-2009, especially 1993-1995。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Interaction chronology, bypass detection and evidence sufficiency.

### CI-0173 — A better validation average can be real while being owned by the wrong path

【时间 / 版本】

V115 epoch-one validation and V116 planning, 2026-07-29.

【当时在解决什么问题】

V115 achieved an unusually low epoch-one deployment action RMSE despite clear
W/FutureEffect failure. The question was whether the result was an illusion and
whether the structural repair should discard it.

【当时的 insight / hypothesis】

A metric improvement can be numerically real but causally misattributed. The
correct response is to preserve the useful behavior while transferring
ownership away from the shortcut, then prove the new owner with interventions.

【具体做法】

- Compared the same epoch, batch size, validation protocol and normalizer across
  V98/V104/V107/V112/V113/V114/V115.
- Decomposed error into horizons and arm/gripper channels.
- Classified V115 as a shortcut baseline for later zero/shuffle experiments.
- Reordered the V116 acceptance criteria to protect temporal/history gains while
  cutting the P3 effect lane's P2/goal/basis bypass.

【观察到的结果】

V115 epoch-one action RMSE was about `0.09957`, roughly `6.55%` below the mean
of the six compared predecessors. About `83.6%` of the MSE reduction came from
steps 13-24; steps 1-4 were slightly worse than V114. Both arm and gripper means
improved, while event/proposal behavior still showed concerns. Existing source
and logs attributed the gain more plausibly to a strong P3/history action route
plus weak auxiliary pressure than to functioning W.

【当时的解释】

Performance preservation and semantic repair are simultaneous requirements.
Deleting the shortcut without preserving its legitimate temporal computation
would confuse ownership repair with capability removal.

【后来怎么看（截至冻结终点）】

Retained. V116 was implemented and statically reviewed, but its claimed closure
still required batch-eight, long-run and frozen-checkpoint intervention evidence.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；V115 performance is historical evidence, not current correctness.

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 2023-2054。
- 段级工件：见 Segment 019 boundary 的 `Primary historical artifacts`。

【关联】

Shortcut baselines, horizon decomposition and capability-preserving repair.

## Segment 019 synthesis

V113-V116 transformed the investigation from module-level activity monitoring
into action-side causal ownership. The strongest probe result was not that the
whole top was useless: G3 address and temporal organization were highly useful,
while the realized W delivery was harmful. Source tracing explained the split:
the future objective could train one projection while P consumed another hidden
carrier, and later V115 could achieve a real epoch-one RMSE gain through a
P3/history shortcut even as W collapsed. The corresponding engineering failures
showed the same ownership problem in a different form: implicit broadcasting and
dense intermediate materialization hid axis and lifecycle semantics. The durable
closure criterion became stricter: preserve the evidence and useful computation,
put supervision on the exact field consumed by policy, keep its axes until the
terminal operation, and verify both action causality and production memory rather
than accepting a label, a loss curve or a passing isolated preflight.

## Segment 020 — V116-V119 window effects, intent identifiability and the top-issue-ledger turn

### Segment boundary

```text
Date:                 2026-07-30
Visible messages:     2055-2380
Starts with:          V116 smoke/long-run launch and comparative evidence
Ends with:            first V119 grounded-intent-effect log and missing new diagnostics
Primary historical artifacts:
  complete V113, V114, V115 and growing V116-V118 logs
  clearvla/policy/goal_conditioning.py
  clearvla/policy/differential_intent_effect.py
  clearvla/policy/grounded_intent_effect.py
  docs/research/TOP_ARCHITECTURE_ISSUE_LEDGER.md
  V116-V119 launchers, manifests, losses and probes
```

### CI-0174 — Lower memory can be evidence of better factorization rather than missing capacity

【时间 / 版本】

V116 batch-eight smoke, 2026-07-30.

【当时在解决什么问题】

V116 used visibly less GPU memory than V114/V115, raising concern that a future
or precision path had silently disappeared.

【当时的 insight / hypothesis】

Memory comparisons first require equal batch/lifecycle identity. Once equal,
lower peak memory is legitimate when it comes from streaming supports, deleting
duplicate unsupervised carriers and narrowing projections around protected
facts—not from reducing evidence axes or query support.

【具体做法】

- Checked whether the observed run was the default batch-one smoke or batch
  eight.
- Traced Teacher-G from simultaneous support buffers to streaming aggregation.
- Checked that the previous 6 GiB Cartesian hidden intermediate remained gone.
- Traced removal of duplicated `state_innovation` and the narrower W decoder.
- Verified that current facts were kept as a protected reference rather than
  concatenated repeatedly into the predicted effect.

【观察到的结果】

V116 could use less memory while retaining future supports and high-resolution
P1 evidence. The reduction was primarily an evaluation/lifetime improvement and
removal of a redundant carrier.

【当时的解释】

Resource health must be explained by a module/axis inventory. Neither a lower nor
a higher peak is self-interpreting.

【后来怎么看（截至冻结终点）】

Retained as a positive example, subject to production lifecycle measurement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 2055-2061。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`state_innovation`。

【关联】

CI-0156, CI-0169 and batch/lifecycle identity.

### CI-0175 — V113/V114/V115 separated stability, engineering efficiency and causal correctness

【时间 / 版本】

Complete three-run replay, 2026-07-30.

【当时在解决什么问题】

Early excerpts had encouraged a single ranking of V113-V115, despite each
version changing different parts of the graph and having different amounts of
complete training evidence.

【当时的 insight / hypothesis】

Versions should be compared on separate axes: experiment identity, action
performance, numerical stability, resource efficiency, representation health
and causal ownership.

【具体做法】

- Verified batch size, seed, data split, normalizer, initialization and epoch
  completeness from serialized context.
- Replayed all available epochs rather than one aligned point.
- Compared gradients, validation bands, gripper/event behavior, W interventions,
  flow use and throughput.

【观察到的结果】

- V113 was the most stable and causally measured of the three, but its W public
  residual was harmful and learned flow barely affected action; throughput was
  about `4.41 s/batch`.
- V114 was a successful engineering optimization at about `2.03 s/batch`, but
  weakened P1/P2 detail delivery and ended slightly worse than V113.
- V115 had real early/gripper and long-horizon gains, but W FutureEffect, phase
  and history mechanisms each degenerated; it had only six complete epochs plus
  a partial seventh.

【当时的解释】

No single version was “the healthy winner.” One supplied stable causal evidence,
one supplied an efficiency pattern, and one supplied a useful shortcut behavior
that later repairs needed to preserve under better ownership.

【后来怎么看（截至冻结终点）】

Retained as a multi-anchor comparison method.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预
- 对话：visible messages 2062-2069。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`4.41 s/batch`、`2.03 s/batch`。

【关联】

Performance anchors, health anchors and experiment identity.

### CI-0176 — W should model observable-window soft futures; S owns cross-window intent

【时间 / 版本】

V116-to-V117 role redesign, 2026-07-30.

【当时在解决什么问题】

W1/W2 repeated the same four FutureEffect predictions. One proposal then pushed
W2 to recursively extrapolate beyond the available 48-frame training window,
which weakened its identifiable supervision.

【当时的 insight / hypothesis】

The user corrected the role boundary: W should model only soft futures supported
inside the observed training window. Cross-window goal progress and remote
control belong to a stateless S, not to an unverifiable W extrapolation.

【具体做法】

- Proposed W1 near and mid successors and W2 a later window consequence.
- Kept each effect spatial/object structured rather than a global intent vector.
- Made S a separate control plane for goal progress, horizon interpretation and
  P temporal organization.
- Rejected the preceding suggestion that W2 query `64-128` frames without a
  corresponding observable teacher window.

【观察到的结果】

The working role map became:

```text
G: current fact and coordinates
W: soft consequences inside the supported future window
S: stateless cross-window intent/control
P: compile precise current facts, window effects and remote intent into action
```

【当时的解释】

An architecture should not assign a module a target that the dataset and teacher
cannot identify merely because the name “world model” suggests arbitrary range.

【后来怎么看（截至冻结终点）】

Retained as a durable ownership distinction; exact slot counts were later
revised.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留原则，三槽 realization later superseded.

【证据与锚点】

- 类型：过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 2074-2094。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Future support, S/W ownership and data-identifiable horizons.

### CI-0177 — A large current-reference loss can disguise nearly absent W supervision

【时间 / 版本】

V116 early log and source audit, 2026-07-30.

【当时在解决什么问题】

FutureEffect losses were nonzero, yet W1/W2 predictions remained highly public
and their gradients were weak.

【当时的 insight / hypothesis】

The loss ledger must be decomposed by parameter ownership. A term named
`W current loss` did not train W innovation when current content came directly
from a G3 projection.

【具体做法】

- Separated current-reference reconstruction from successor, semantic,
  transport, covariance and status targets.
- Traced which terms depended on W route parameters.
- Examined the common reliability multiplier applied to otherwise distinct
  target types.
- Compared W1/W2 objectives and found that their functional definitions were
  largely repeated rather than proposal/refinement specific.

【观察到的结果】

Teacher reliability was around `0.03`; the genuinely differentiating W targets
were therefore strongly attenuated, while the common current target stayed easy
and full-strength. W1/W2 successor losses were almost identical and prediction
cosine approached `0.997`.

【当时的解释】

An aggregate auxiliary loss can look substantial while nearly all of its usable
gradient belongs to an upstream reference head. Reliability for identity,
transport and visibility also should not be assumed interchangeable.

【后来怎么看（截至冻结终点）】

Retained. Later teacher versions changed the exact reliability algebra, but the
parameter-ownership audit remains valid.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 2070-2084 and 2095-2103。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Loss ownership, W publicization and target reliability.

### CI-0178 — Repeated probability filtering can be forward-finite and backward-catastrophic

【时间 / 版本】

V116 epoch two failure, 2026-07-30.

【当时在解决什么问题】

V116 terminated at epoch 2 batch 2631 with finite forward losses followed by
NaN gradients across hundreds of parameters. The failure report printed proposal
parameters first, tempting a false source attribution.

【当时的 insight / hypothesis】

A normalized probability recurrence can remain finite in value while its
Jacobian becomes pathological. Parameter traversal order identifies the first
reported victim, not necessarily the upstream source.

【具体做法】

- Traced the four-state phase belief through nine repeated
  `propagate * observation_evidence / clamp(sum)` operations.
- Examined phase mass, max probability and transition gradients before failure.
- Distinguished the first registered `proposal.*` parameters from the causal
  origin of a graph-wide NaN.
- Replaced the recurrence in the next design rather than clipping the resulting
  gradients locally.

【观察到的结果】

The phase recurrence lost healthy normalization/conditioning, produced abnormal
phase statistics and then contaminated at least 573 parameter tensors during
backward, although the scalar loss was finite immediately beforehand.

【当时的解释】

Forward boundedness is not a sufficient numerical contract. Repeated divisions,
probability products and low-mass normalization require Jacobian-level tests and
first-nonfinite localization.

【后来怎么看（截至冻结终点）】

Retained as a likely buried source of the delayed-gradient-explosion family. The
exact historical implementation was later replaced by S, not rescued through
clipping.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V116 recurrence superseded.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 2126-2132。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`propagate * observation_evidence / clamp(sum)`。

【关联】

CI-0159, phase recurrences and non-finite sentinels.

### CI-0179 — A scalar progress coordinate is a diagnostic projection, not a task-state owner

【时间 / 版本】

V117 S design and audit logging, 2026-07-30.

【当时在解决什么问题】

The replacement S exposed a continuous progress value, but the user objected
that one scalar conflates retries, pauses, overlapping operations and distinct
states at the same trajectory fraction.

【当时的 insight / hypothesis】

Stage should be a structured, re-inferred intent state. A scalar may summarize
that state for audit, but it must not globally modulate G/W/P or substitute for
“what should happen now.”

【具体做法】

- Added audit-only `frame_progress` from the sample's normalized episode index.
- Logged `intent_progress`, signed gap and MAE under detach/no-grad.
- Kept frame position out of model input, attention, control, loss, checkpoint
  and deployment.
- Proposed correlations, quantiles and progress bins for later validation rather
  than interpreting individual batch means.

【观察到的结果】

Early V117 examples placed intent progress near `0.5` despite changing frame
progress. Later evidence showed poor or negative correlation, consistent with a
functional constant rather than learned stage.

【当时的解释】

Trajectory position is useful as a shortcut detector, not as automatic phase
ground truth. A model should earn stage utility by improving matched future and
action organization.

【后来怎么看（截至冻结终点）】

Retained. Later plans removed scalar progress from routing while preserving it as
a diagnostic projection.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；方案/设计记录
- 对话：visible messages 2134-2147 and 2174-2179。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`frame_progress`、`intent_progress`。

【关联】

Stateless intent, phase identifiability and audit-only labels.

### CI-0180 — Runtime contracts became a replicated failure surface

【时间 / 版本】

V117 startup and contract discussion, 2026-07-30.

【当时在解决什么问题】

A teacher field was named `reliability_slots` at one boundary and
`reliability_target_slots` at another. Producer, loss, preflight, tests and
version validators each repeated the same string protocol.

【当时的 insight / hypothesis】

Contracts are useful when they centralize semantics; they become side effects
when they duplicate the interface across more places than the underlying data
path. Version identity, feature switches, tensor keys and numerical invariants
had been conflated.

【具体做法】

- Repaired the immediate field name at producer and two consumers.
- Distinguished typed data interfaces, module-boundary invariants and outer
  experiment configuration.
- Proposed keeping `vxxx` only as an experiment/checkpoint label.
- Proposed a compatibility adapter that maps old CLI state once into named
  capabilities, after which the model graph contains no version branches.
- Preserved useful checks: shapes, dtype, device, finite values, teacher
  isolation, five-step lifecycle, optimizer coverage and resume identity.

【观察到的结果】

The immediate regression passed, but the discussion established that a passing
version validator did not prove causal closure and could itself prevent a run
from starting after a semantically harmless rename.

【当时的解释】

The remedy is not “no contract”; it is one authoritative typed producer-consumer
boundary plus independent experiment and lifecycle checks.

【后来怎么看（截至冻结终点）】

Retained. The user explicitly rejected continued internal organization around
`Vxxx`, while also rejecting an aggressive rewrite that would destroy experiment
continuity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；方案/设计记录
- 对话：visible messages 2148-2162。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`reliability_slots`、`reliability_target_slots`。

【关联】

Typed interfaces, capability identities and research-system refactoring.

### CI-0181 — Representational differentiation without an indispensable task is not stage learning

【时间 / 版本】

V117 early-to-mid training, 2026-07-30.

【当时在解决什么问题】

V117 S had multiple blocks, nonzero gradients and somewhat different internal
program states, yet window selection stayed nearly uniform and progress stayed
near a constant.

【当时的 insight / hypothesis】

Providing capacity to represent stage does not make stage identifiable. If W
slots are public, P2 has a fixed temporal prior and P3/history can organize
action directly, the optimal S can remain an average state.

【具体做法】

- Separated phase belief, active/next/remaining intent and W-window selector in
  the logs rather than calling all three “stage.”
- Compared S1/S2/S3/MLP gradients to actual selector entropy and window output
  cosine.
- Aligned their trajectories with W publicization and P3 lane use.
- Refused to turn frame progress into a direct loss or impose entropy/quota
  targets.

【观察到的结果】

S blocks received growing gradients and phase belief was not exactly uniform,
but selector outputs were nearly uniform, intent progress remained near `0.5`
and the downstream windows were almost identical. Action optimization remained
healthy because other paths supplied temporal organization.

【当时的解释】

Stage becomes functional only when its state conditions a supervised effect that
the action consumer cannot replace, and matched S interventions first alter that
effect before action.

【后来怎么看（截至冻结终点）】

Retained. V117 showed why “more blocks plus heads” was not closure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V117 S superseded.

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 2163-2203。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Identifiability, credit assignment and audit-only progress.

### CI-0182 — Shared targets, fixed priors and easy public consumers form one publicization chain

【时间 / 版本】

V117 source/log synthesis and V118 planning, 2026-07-30.

【当时在解决什么问题】

W's three effect slots converged to a common direction; S had different internal
content but nearly identical window reads; P2 and P3 remained active.

【当时的 insight / hypothesis】

The observations were not six unrelated weak modules. They formed a credit-
assignment chain:

1. the full-strength current target was shared across windows;
2. distinguishing targets were reliability-attenuated;
3. S window queries read almost the same program mixture;
4. P2 could rely on a fixed near/mid/late temporal prior;
5. P3 history/temporal paths could bypass effect.

【具体做法】

- Traced the common current expansion and parameter ownership.
- Compared target interval difference with prediction cosine/variation.
- Traced P2 key construction and temporal priors.
- Traced P3 effect/temporal inputs and null behavior.
- Drafted the differential intent-effect repair as a synchronized ownership
  change rather than a larger S alone.

【观察到的结果】

V117 could continue lowering action loss while W prediction cosine remained
near one, selector entropy remained high and P3 preferred history/temporal.
This established a structural optimum, not merely slow learning.

【当时的解释】

If the common component is stronger and all difference-bearing consumers are
optional, every local optimizer receives the same incentive to publicize.

【后来怎么看（截至冻结终点）】

Retained. V119, V120 and later schema audits repeatedly found the same broader
publicization chain, confirming that V118 strengthened one route without closing
all of its target, prior and consumer members.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2166-2218。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Common targets, reliability, fixed priors and optional consequence.

### CI-0183 — AttnRes should select formed innovations; it cannot create missing identities

【时间 / 版本】

V117/V118 AttnRes reconsideration, 2026-07-30.

【当时在解决什么问题】

The team reconsidered Kimi Attention Residuals as a remedy for W/S
publicization.

【当时的 insight / hypothesis】

AttnRes is valuable for choosing which residual history to inherit, but it
cannot recover an object, type or window distinction that was never formed or
was already summed away. Protected common facts must remain outside selection;
only genuine typed/window innovations should compete.

【具体做法】

- Re-read Attention Residuals as a residual-history mechanism rather than a
  generic attention upgrade.
- Applied its boundary principle to `protected_current_G + selected_local_delta`.
- Rejected routing cumulative W hidden or repeated current content.
- Noted that S/W roles must already differ before any selector is meaningful.

【观察到的结果】

The principle clarified why adding attention to the common carrier would not
solve V118. Later source review found W was slot-blind before the relevant
selection boundary, so the missing identity problem was upstream of AttnRes.

【当时的解释】

Routing is a consumer operation, not an identity generator.

【后来怎么看（截至冻结终点）】

Retained. Exact historical AttnRes variants are not current authority.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 2265-2274。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Protected bases, typed deltas and selector prerequisites.

### CI-0184 — Near-one diagnostics were different failure modes hidden by imprecise names

【时间 / 版本】

V118 early-log audit, 2026-07-30.

【当时在解决什么问题】

Several V118 metrics printed as `1`, creating ambiguity over whether the model
had selected stage one with certainty or whether multiple states were identical.

【当时的 insight / hypothesis】

An argmax index, an entropy, a cosine and a probability mass must never share
selection-like naming. Rounded cosines also do not prove equality, and cosine
ignores magnitude.

【具体做法】

- Traced `intent_*_program` to a detached argmax index averaged across batch.
- Converted normalized attention entropy `0.993` over four tokens to an
  effective state count of about `3.96`.
- Distinguished `intent_window_cosine≈1` from
  `w1_effect_cosine≈1` and from an actual probability.
- Proposed logging max mass, entropy, pairwise attention L1, exact argmax name
  and per-head rather than head-averaged distributions.

【观察到的结果】

All three windows printed program index 1 because the second token won by a tiny
amount under an almost uniform posterior. The windows and W effects were nearly
collinear, but not necessarily equal in amplitude.

【当时的解释】

Metric schema is part of the semantic contract. Misnamed diagnostics can make
uncertainty look like certainty and public direction look like exact tensor
identity.

【后来怎么看（截至冻结终点）】

Retained as a logging rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；方案/设计记录
- 对话：visible messages 2268-2285。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Metric provenance, entropy interpretation and publicization.

### CI-0185 — V118 exposed four independent ways to bypass a nominal W midpoint

【时间 / 版本】

V118 complete top-down source audit, 2026-07-30.

【当时在解决什么问题】

The code invoked W and routed a `FutureEffect` object into P2, yet W still failed
to behave as the causal midpoint for future consequences.

【当时的 insight / hypothesis】

Structural order is not causal necessity. A future path can be bypassed through
loss-free content, non-neutral defaults, a free organizer or an unbound temporal
lane even when every named module is called.

【具体做法】

The source review found:

- W effect decode was slot-blind: a route without object-slot identity was
  copied to all slots.
- P2 treated covariance, visibility, persistence and uncertainty defaults as
  nonzero effect values even when semantic effect was neutral.
- The consequence organizer jointly consumed factual base, effect base and P2
  delta, so an ordinary MLP could ignore or cancel effect.
- P3 temporal consumed temporal control and basis without the consequence it was
  supposed to schedule.
- S window outputs discarded their own typed query bases.
- The intent teacher averaged camera, space and slot, destroying local changes.

【观察到的结果】

W prediction cosine approached `0.99` and variance approached zero while the
P2 reader became strong. Consequence organizer RMS was about four times the
effect base. Action loss could fall without object-specific future effects.

【当时的解释】

For a causal midpoint, neutral output must be algebraic zero, its identity axes
must survive producer to consumer, and the consequence consumer must not have a
same-purpose input that substitutes for it.

【后来怎么看（截至冻结终点）】

Retained as the source basis for the later grounded-intent-effect reset.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V118 implementation superseded.

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2275-2307。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。

【关联】

Object-slot identity, neutral defaults, consequence closure and temporal binding.

### CI-0186 — Problem ledgers and architecture contracts serve different memory functions

【时间 / 版本】

V118 top audit, 2026-07-30.

【当时在解决什么问题】

The first review accumulated current defects inside an already large architecture
contract. The user objected that this mixed diagnosis, desired design and
historical narrative, making unresolved problems hard to close.

【当时的 insight / hypothesis】

A current problem ledger should contain only unresolved defects, evidence,
impact, dependencies, missing validation and closure conditions. The architecture
contract should describe the accepted executable graph. Historical changes
belong elsewhere.

【具体做法】

- Limited the audit to Pre-G/G/S/W/P and direct teacher/loss/probe boundaries.
- Created `TOP_ARCHITECTURE_ISSUE_LEDGER.md` with severity and evidence fields.
- Recorded source/log connections and dependency ordering.
- Excluded bottom from that audit at the user's explicit request.
- Required issues to be removed or archived when actually closed.

【观察到的结果】

The ledger made it possible to distinguish T01-T10 blocking/high-priority
architecture defects, T11 semantic ambiguity and T12-T15 observability/test
blind spots. The user twice corrected reviews that had summarized conclusions
without actually recording or correlating all available metrics.

【当时的解释】

Documentation is part of the review loop: if unresolved evidence cannot be found
independently of the desired architecture story, later revisions will repeat it.

【后来怎么看（截至冻结终点）】

Retained. This interaction materially tightened the requirement for source-plus-
log accounting before planning.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 2289-2327。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`TOP_ARCHITECTURE_ISSUE_LEDGER.md`。

【关联】

Repository memory, issue closure and interaction-driven review changes.

### CI-0187 — V119 corrected false stages by grounding intent in observable types and four real intervals

【时间 / 版本】

V118-to-V119 redirect, 2026-07-30.

【当时在解决什么问题】

The differential-intent plan treated four learned T5 queries as ordered stages
without stage labels, compressed four teacher intervals into three slots and
fused typed evidence before its owners were formed.

【当时的 insight / hypothesis】

The plan itself was partially wrong. S should organize observable goal/history/
typed evidence without claiming latent phase classification; W should retain the
teacher's actual four intervals; object/camera/space/type axes must traverse
G-to-Teacher-to-W-to-P2; and neutral future must reduce consequence exactly to
the P1 fact.

【具体做法】

- Replaced scalar/ordered-stage assumptions with a stateless intent organizer.
- Restored `4-8 / 8-16 / 16-32 / 32-48` future intervals.
- Preserved G2-to-G3 ownership and object/camera/space/type axes.
- Made the supervised `FutureEffectField` the object consumed by P2.
- Made P2 routing bounded and zero-centered.
- Required P3 precision/temporal to use the consequence state.
- Kept Teacher loss-only and static P1 read once per observation.

【观察到的结果】

The resulting `grounded_intent_effect_323` implementation was reported to pass
local tests and reject incompatible top checkpoints. The first V119 log showed
new-path gradients but omitted all planned V119 ground/intent/effect/policy
diagnostics and still used V114 log labels.

【当时的解释】

The semantic reset addressed clear source defects, but missing observability
meant the run could not yet prove that the realized graph satisfied the plan.

【后来怎么看（截至冻结终点）】

Partly retained. The complete V119 audit confirmed active grounded computation
but still found weak/marginal W action use and unresolved object ownership.
Observable typed intent and four real intervals were carried into V120; the V119
realization was not retained as a healthy endpoint.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：被后续 V120 吸收；V119 healthy-endpoint claim rejected.

【证据与锚点】

- 类型：日志观测；测试/运行观测；方案/设计记录；推断/未决
- 对话：visible messages 2302-2380。
- 段级工件：见 Segment 020 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`grounded_intent_effect_323`。
- 直接量值/表达式（卡内原文）：`4-8 / 8-16 / 16-32 / 32-48`。

【关联】

False stage labels, four intervals, object-axis continuity and diagnostic closure.

## Segment 020 synthesis

V116-V119 showed why a series of individually reasonable repairs kept producing
new public shortcuts. V116 narrowed the W carrier and improved memory/action
optimization, but its repeated phase probability recurrence was numerically
ill-conditioned and its future loss mostly belonged to a protected current
reference. V117 replaced recurrence with a richer stateless S, yet capacity and
nonzero gradients could not identify stage while fixed priors and P3/history
made it optional. V118 made S/W/P more explicit but still lost object identity,
used non-neutral default effects and allowed consequence/temporal bypasses. The
resulting methodological correction was as important as the next architecture:
audit the whole top graph, keep an independent unresolved-issue ledger, treat
AttnRes as a selector of already formed innovations, and revise the plan itself
when its stages or intervals are not supported by data. V119 began from that
reset, but its first log immediately demonstrated that implementation and
observability must close together.

## Segment 021 — V119 causal measurement, V120 object semantics and completion removal

### Segment boundary

```text
Date:                 2026-08-01
Visible messages:     2381-2652
Starts with:          complete eight-epoch V117/V118/V119 comparison
Ends with:            first completion-off/schema-2 V120 log handoff
Primary historical artifacts:
  complete V117, V118 and V119 logs
  V119 frozen model-path probe and evaluator
  clearvla/policy/object_intent_dynamics_323/
  object-intent-dynamics V120 logs, smoke and manifests
  docs/research/TOP_ARCHITECTURE_ISSUE_LEDGER.md
```

### CI-0188 — Better future auxiliary loss can coexist with worse action generalization

【时间 / 版本】

Complete V117/V118/V119 replay, 2026-08-01.

【当时在解决什么问题】

V118 had stronger explicit S/W/P connections and lower training/future-content
loss, yet did not look better on validation. V119 appeared slightly stronger but
omitted its new structural metrics.

【当时的 insight / hypothesis】

An auxiliary objective can improve while its representation is routed through
the wrong axes, used as a shortcut or ignored by the action consumer. Validation
and causal action utility must therefore be compared alongside representation
loss, not inferred from it.

【具体做法】

- Verified identical data, seed, batch, Stage1 setting, normalizer and training
  length across the three completed runs.
- Compared every epoch and horizon/channel metrics rather than a best point.
- Reused the existing V117/V118 source conclusions and treated V119's missing
  diagnostics as a separate observability defect.

【观察到的结果】

At epoch three, V117 action RMSE was about `0.08252` and V118 about `0.08956`
(`8.5%` worse), while V118 future content was about `23.6%` better. V118
gripper RMSE was about `18.9%` worse and event F1 lower. V117 remained the
smoothest stable baseline; V119 slightly improved its best validation point but
could not yet prove Grounded G/S/W/P use.

【当时的解释】

The loss was training an easier or mis-owned property of the future state. The
consumer contract, not mere loss decrease, determines policy value.

【后来怎么看（截至冻结终点）】

Retained as a central multi-objective evaluation rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2382-2398。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

CI-0162, representation/action decoupling and full-epoch comparison.

### CI-0189 — Active computation and diagnostic registration can fail independently

【时间 / 版本】

V119 complete-run audit, 2026-08-01.

【当时在解决什么问题】

Grounded losses and gradients were present, but the log still used V114 labels
and omitted all planned `v119-ground/intent/effect/policy` rows.

【当时的 insight / hypothesis】

The model can train through one explicit loss/gradient route while its forward
diagnostics are filtered by an obsolete scalar whitelist. Absence from the log
does not prove absence from the graph; conversely, active gradients do not make
an unobserved structure healthy.

【具体做法】

- Traced Grounded diagnostics to `output["grounded_*"]`.
- Traced loss construction and post-backward gradient logging through separate
  paths.
- Repaired metric registration, active markers and V119 label selection.
- Split W gradient groups and made W2 statistics late-only.
- Reduced the current issue ledger from a large V118 history stack to unresolved
  V119 questions.

【观察到的结果】

Existing logs still showed useful trends: moving-warp gain improved, late-detail
ratio grew, P2 reader gradients increased and P3 precision/temporal were more
balanced. But W action utility still required a frozen intervention.

【当时的解释】

Observability is a first-class consumer of the interface, with its own identity
and lifecycle tests.

【后来怎么看（截至冻结终点）】

Retained. This also corrected misleading W1/W2 gradient group comparisons.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录；方案/设计记录
- 对话：visible messages 2388-2410。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`v119-ground/intent/effect/policy`。

【关联】

Metric registration, active manifests and issue-ledger hygiene.

### CI-0190 — Experiment paths must distinguish raw data, derived caches and weights

【时间 / 版本】

V119 smoke/probe environment repair, 2026-08-01.

【当时在解决什么问题】

An instruction that weights and data were under `/data/senwang` was interpreted
too broadly, changing the raw HDF5 root to a directory that contained only
derived caches.

【当时的 insight / hypothesis】

Storage roots are typed resources. A cache migration does not imply raw data,
language embeddings or initialization checkpoints migrated with it. Defaults
must be verified against actual files and the active Stage1 setting.

【具体做法】

- Located the 73 HDF5 episodes at the existing Liang Zhang dataset root.
- Kept decoded and DINO caches under `/data/senwang/data`.
- Verified the T5 embedding under `/data/senwang/checkpoint`.
- Removed an incorrectly added `best_contract.pt` default because Stage1
  initialization was disabled.

【观察到的结果】

The smoke proceeded after restoring the raw-data path. The incident prevented a
second false attribution to model code or T5 conditioning.

【当时的解释】

Launcher provenance is part of experiment identity; path names cannot substitute
for checking resource contents and consumer roles.

【后来怎么看（截至冻结终点）】

Retained as operational discipline.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测；数据探针；推断/未决
- 对话：visible messages 2411-2425。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`/data/senwang`、`/data/senwang/data`、`/data/senwang/checkpoint`、`best_contract.pt`。

【关联】

Run identity, caches and initialization provenance.

### CI-0191 — A capability validator must not replay incompatible ancestry assumptions

【时间 / 版本】

V119 evaluator repair, 2026-08-01.

【当时在解决什么问题】

The Grounded checkpoint was rejected as missing a V111 contract because the
evaluator recursively required an old fixed-address loss that the new capability
had intentionally disabled.

【当时的 insight / hypothesis】

Compatibility should validate the active capability graph directly. Replaying
every ancestor's incidental configuration turns historical implementation
choices into false invariants.

【具体做法】

- Routed `auto` and explicit Grounded identities directly to the Grounded
  validator.
- Kept full current-graph checks without requiring V105-V111 loss settings.
- Unified optional probe enablement and summary counts under the same boundary,
  so intentionally skipped V113-only checks did not later fail coverage.

【观察到的结果】

The existing V119 checkpoint became evaluable without retraining or relaxing its
current contract. Two independent ancestry/counting errors were removed.

【当时的解释】

An exact resume identity and an architecture ancestry story are different
objects. Only active invariants belong in capability validation.

【后来怎么看（截至冻结终点）】

Retained; this foreshadowed later schema-based manifests.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；历史检索
- 对话：visible messages 2426-2442。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

CI-0180, manifest schemas and evaluation compatibility.

### CI-0192 — A causal probe is invalid when the explicit baseline changes the forward graph

【时间 / 版本】

First V119 frozen probe, 2026-08-01.

【当时在解决什么问题】

Many intervention rows reported “pass,” but ordinary deployment and an explicit
`none` intervention differed by `9.842e-3` under the same input and fixed noise.

【当时的 insight / hypothesis】

Intervention effects smaller than baseline execution drift have no causal
meaning. A probe's no-op path must be bit-equivalent to production or fail before
producing acceptance judgments.

【具体做法】

- Traced explicit `none` to an audit execution branch and raw-reader `measure`
  mode.
- Found `need_weights=True/False` selected different BF16 attention kernels.
- Found the acceptance matrix treated any metric name containing `delta` as
  evidence of intervention reach, including natural G delta norms.
- Moved detached attention diagnostics to an FP32 side computation.
- Required per-batch replay max error `<=1e-8` and mode-specific boundary
  whitelists.
- Moved language interventions to the real pre-S T5 boundary.

【观察到的结果】

The corrected probe achieved exact replay on all four selected batches. Previous
W changes of roughly `0.05%-0.2%` could no longer be accepted merely because a
confidence interval excluded zero under the drifting baseline.

【当时的解释】

A measurement tool is part of the executable graph. Its kernel selection,
diagnostic side effects and acceptance algebra require the same rigor as model
code.

【后来怎么看（截至冻结终点）】

Retained. The user also imposed a useful stopping rule: repair the measurement
once, then stop expanding probes when source defects are already decisive.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；过程/决策记录
- 对话：visible messages 2443-2461 and 2486-2505。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`need_weights=True/False`。

【关联】

CI-0167, exact replay and probe scope.

### CI-0193 — V119's working policy was P1 detail plus P3 precision/temporal, not W

【时间 / 版本】

Corrected V119 frozen probe, 2026-08-01.

【当时在解决什么问题】

The team needed a functional map of the trained checkpoint after the probe itself
became trustworthy.

【当时的 insight / hypothesis】

Large matched effects can establish a working route; tiny W effects should be
reported as weak rather than inflated into proof of future modeling.

【具体做法】

- Used 32 samples from four of five validation episode clusters with exact
  baseline replay.
- Zeroed/shuffled protected detail, literal RGB, whole S, P3 precision and P3
  temporal at their realized boundaries.
- Logged FutureEffect before mask, after validity and after reliability.

【观察到的结果】

- Protected-detail shuffle worsened MSE about `49.27%`.
- P3 precision zero worsened it about `11.96%`; shuffle worsened it `0.79%`.
- Whole intent-state zero worsened it about `8.06%`.
- P3 temporal zero worsened it about `4.51%`, primarily at horizons 5-24.
- Literal RGB zero worsened it about `1.83%`.
- FutureEffect RMS was `0.025554` before masking, unchanged by validity, but only
  `0.003834` after reliability: about `15%` remained.

【当时的解释】

The reliable mainline was current high-resolution evidence organized by P3,
with useful S/temporal support. W was syntactically connected but its value had
been attenuated by about `85%` before P2.

【后来怎么看（截至冻结终点）】

Retained as the functional baseline that motivated V120.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 2462-2491。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

P1 factual path, P3 lane ownership and reliability attenuation.

### CI-0194 — Global object slots require cross-slot allocation, not four independent reads

【时间 / 版本】

V119 G source/probe audit, 2026-08-01.

【当时在解决什么问题】

V119 reported four effective slots but their centers, posteriors and policy
queries were nearly identical.

【当时的 insight / hypothesis】

Four slots independently softmaxing over the same candidate library do not form
four objects. Without candidate-to-slot allocation or cross-slot competition,
the symmetric solution is structurally natural; a small slot embedding and a
late slot softmax cannot reliably break it.

【具体做法】

- Traced G1 candidate softmax separately for each slot.
- Traced G2's late slot evidence softmax after independent candidate search.
- Traced G3 as a bounded residual incapable of easily separating an already
  symmetric state.
- Compared source expectations with exact probe statistics.

【观察到的结果】

Slot center distance was about `0.00306`, posterior Hellinger `0.00607`, slot
entropy `0.999917`, effective slots `3.99954` and policy-query variation
`0.00101`. Effective count near four meant four nearly identical slots were
uniformly used, not four objects were healthy.

【当时的解释】

Object identity is a global assignment problem. Dense spatial evidence and
global object slots should remain separate interfaces.

【后来怎么看（截至冻结终点）】

Retained; V120 introduced a dense chart plus global K-object binder.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 2492-2515。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

Slot Attention, dense charts and global object identity.

### CI-0195 — V120 reset the top around object, intent and dynamics ownership

【时间 / 版本】

Object-Intent-Dynamics 3-2-3 design and implementation, 2026-08-01.

【当时在解决什么问题】

G-slot homogeneity, reliability-suppressed W, proposal conditioning that ignored
current goal/world state and S/P1 routes that made W optional had survived
several incremental plans.

【当时的 insight / hypothesis】

The top needed a coherent ownership reset rather than another V119 patch:

- a dense chart preserves all spatial evidence;
- global K objects compete over that chart;
- S produces intent queries rather than freely decodable action value;
- an online coarse action intent conditions W;
- W predicts four object-level future dynamics intervals;
- P1 keeps precise current facts, P2 reads effects with exact-zero semantics and
  P3 exposes typed lanes;
- generic parallel W/P/bottom aliases are closed.

【具体做法】

Implemented capability `object_intent_dynamics_323` while keeping V120 only as
an experiment label. During review, replaced zero-point cosine normalization
with a bounded alternative and changed W2 from mean-compressing W1 near intervals
to cross-attending both.

【观察到的结果】

The implementation passed local static/test review and preserved Teacher
train-only and static/dynamic lifecycle boundaries. Server smoke then exposed
several implementation/lifecycle errors before producing usable early logs.

【当时的解释】

This version was intended to remove the common cause rather than force W use by
gain or loss. Its success still depended on validating the new semantic
interfaces in production.

【后来怎么看（截至冻结终点）】

Retained as a major architecture pivot and the source of later schema work, not
as evidence that its first implementation was correct.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收并多次修订。

【证据与锚点】

- 类型：日志观测；测试/运行观测；方案/设计记录；推断/未决
- 对话：visible messages 2500-2563。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`object_intent_dynamics_323`。

【关联】

G/S/W/P ownership, global K and safe normalization.

### CI-0196 — Preflight must reproduce mode, autocast and snapshot boundaries of production

【时间 / 版本】

V120 startup repairs, 2026-08-01.

【当时在解决什么问题】

Three startup failures appeared in sequence: a training-only future pack was
tested under `eval`, Teacher targets became BF16 despite explicit `.float()`, and
a process imported old and new modules during an in-progress source sync.

【当时的 insight / hypothesis】

Preflight validity depends on the same train/eval mode, local autocast domain and
source snapshot as production. A `.float()` call inside active CUDA autocast is
not sufficient to keep later Linear/einsum computations FP32.

【具体做法】

- Ran teacher/recognizer preflight temporarily in train mode under no-grad and
  restored eval afterward.
- Disabled autocast locally around Teacher association, softmax, moments and
  target construction.
- Split validator messages into actual dtype and finite status.
- Verified a fresh server Python process after all source files finished syncing.

【观察到的结果】

The full smoke completed five-step deploy, teacher-forced training, backward,
validation and probe. Loss accounting closed and gradients remained finite.

【当时的解释】

An isolated check can create both false leakage reports and false precision
confidence if it does not reproduce the real lifecycle.

【后来怎么看（截至冻结终点）】

Retained as a deployment/research-operations invariant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 2566-2593。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

Teacher FP32, no-grad isolation and atomic source synchronization.

### CI-0197 — Allocation, existence, validity and null are not interchangeable probabilities

【时间 / 版本】

V120 epoch-one batch 640-1700, 2026-08-01.

【当时在解决什么问题】

The new binder separated object contents, yet existence fell near `0.10`, null
mass rose near `0.90` and P2 increasingly rejected W effects.

【当时的 insight / hypothesis】

An object owner prior describes how chart mass is allocated among objects. It is
not evidence that the object or camera support is invalid. Turning `1-prior`
into null converted relative competition into physical absence and gave the
model a cheap way to shut off W.

【具体做法】

- Restricted `owner_prior` to local candidate mixture.
- Defined `allocation` as chart share for audit.
- Defined `existence` as object-vs-null confidence on an object's own read.
- Defined `validity` from real candidate/camera support and made it the only
  future loss/W/P2 mask.
- Reserved null for actual unsupported association.
- Added mass-error and separate allocation/existence/validity/null diagnostics.

【观察到的结果】

Before the fix, null reached about `0.896` despite healthy visual support. After
tracing, the error explained why learned object differences were discarded at
W-to-P2 rather than proving W had failed to form them.

【当时的解释】

Probability names encode different sample spaces. Multiplying them together
without preserving their semantics can create an artificial information gate.

【后来怎么看（截至冻结终点）】

Retained as one of the clearest V120 semantic wiring bugs. Later schemas refined
loss support versus selector validity further.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 2594-2619。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`owner_prior`。

【关联】

Null semantics, future support and cheap rejection paths.

### CI-0198 — V120 corrected the claim that all of G/W was public; time remained the weak axis

【时间 / 版本】

Pre-binder-fix V120 through batch 1700, 2026-08-01.

【当时在解决什么问题】

Earlier summaries described the whole G/W path as public, but the longer V120
prefix provided object- and interval-specific statistics.

【当时的 insight / hypothesis】

Object differentiation and temporal differentiation must be measured
independently. A path can preserve object identity while under-representing
future interval change, then lose the useful objects later through a bad mask.

【具体做法】

- Tracked G object-pair cosine and W1/W2 object cosine over the same prefix.
- Compared W interval cosine/variation with teacher interval variation.
- Aligned these with validity/null and P2 effect use.

【观察到的结果】

G object-pair cosine fell from about `0.905` to `0.374`; W1/W2 object cosines
fell similarly to about `0.373/0.371`. W interval cosines remained high at about
`0.991/0.968`, while teacher interval cosine was `0.894`; predicted interval
variation `0.038` was only roughly a third of target variation `0.101`.

【当时的解释】

V120 was learning objects and W was retaining them. The confirmed faults were
the invalid rejection semantics and a weaker temporal bandwidth—not total W
publicization.

【后来怎么看（截至冻结终点）】

Retained as an explicit correction to overbroad earlier language.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 2614-2619。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

Object versus interval axes and evidence-driven conclusion revision.

### CI-0199 — An unsupervised completion scalar became a duplicate execution bias

【时间 / 版本】

V120 completion audit, 2026-08-01.

【当时在解决什么问题】

S completion probability rose near one and then collapsed near zero, behaving
like a global bias rather than sample-dependent task completion.

【当时的 insight / hypothesis】

The desired stateless function was weak modulation from observable state change,
not a second terminal decision. Without completion labels, the scalar could only
learn through action/execution gradients and naturally became a global control
knob that duplicated the bottom executor.

【具体做法】

- Traced completion evidence to both a P3 terminal lane and a direct terminal
  logit bias in the execution controller.
- Preserved S goal/history/object/interval/temporal processing.
- Removed completion probability/uncertainty and the external terminal bias.
- Reused the same fifth P3 lane width and `0.05` budget for a bias-free,
  zero-centered observable state-change vector derived from history difference
  and G transport.
- Required zero history/transport change to produce exact zero.

【观察到的结果】

The replacement left G/W/P1/P2 and the first four P3 lane formulas unchanged;
the original execution controller retained sole terminal ownership. A binder-
fixed but completion-on run showed the same collapse timing, proving the two
faults were independent.

【当时的解释】

If a semantic head has no identifiable target and duplicates an existing
controller, adding an explanatory name does not prevent it from becoming a
global bias.

【后来怎么看（截至冻结终点）】

Retained. The exact state-change implementation still required experimental
validation after this cursor.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；completion branch retired.

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 2620-2652。
- 段级工件：见 Segment 021 boundary 的 `Primary historical artifacts`。

【关联】

Stateless state change, terminal ownership and zero semantics.

## Segment 021 synthesis

This segment converted V119/V120 from architectural narratives into measured
functional maps. Exact-replay probing showed that V119's strongest policy was
the P1-detail/P3-precision route, with useful S and temporal support, while
reliability removed roughly 85% of FutureEffect before P2. Source review then
showed why global slots were symmetric and motivated the Object-Intent-Dynamics
reset. Crucially, longer V120 evidence corrected the overly broad diagnosis that
G/W were wholly public: object representations did separate and W preserved
them, while interval variation remained weak and a semantic wiring error turned
allocation into invalidity/null, discarding the objects at the consumer. The
completion audit exposed a separate duplicate-control shortcut. Across both
faults, the durable rule is to distinguish probability sample spaces and causal
owners precisely: allocation is not validity, uncertainty is not effect
amplitude, and observable state change is not terminal authority.

## Segment 022 — V120 controlled results, typed docking and V121/V122 temporal-path audit

### Segment boundary

```text
Date:                 2026-08-08
Visible messages:     2653-2801
Starts with:          completed V120 three-run comparison
Ends with:            V122 schema-4 gradient-diagnostic startup repair
Primary historical artifacts:
  v120_begin.log, v120_long_pre_fix.log and v120_long.log
  V121/V122 logs
  object_intent_dynamics_323 grounding, intent, dynamics and compiler modules
  Fast-WAM, Faster-WAM/DoT and MV-WAM design discussion
  top architecture issue ledger
```

### CI-0200 — Removing the unsupervised completion bias improved the controlled V120 run

【时间 / 版本】

Three V120 runs, 2026-08-08.

【当时在解决什么问题】

The completion-to-terminal path had been removed on semantic grounds, but the
user required evidence that the local replacement did not damage the rest of
the policy.

【当时的 insight / hypothesis】

Eliminating an unidentifiable duplicate controller can improve generalization
even when it reduces an apparent high-level capability. The correct comparison
is a run with the binder already fixed and completion retained versus the same
graph with zero-centered observable state change.

【具体做法】

- Identified the three logs by manifest/source behavior, not file timestamp.
- Treated the old-binder partial run only as fault evidence.
- Compared the two complete binder-fixed eight-epoch runs across action, horizon,
  arm/gripper, event and late-epoch behavior.

【观察到的结果】

Completion-off/state-change reduced final validation RMSE from about `0.08687`
to `0.08145` (`6.2%`), moved the best point from `0.08370` at epoch four to
`0.07931` at epoch seven, and improved tail, gripper and event-head metrics. A
small epoch-seven-to-eight rebound and weak long-horizon/gripper behavior
remained.

【当时的解释】

The result supported the semantic diagnosis: completion had acted as a harmful
global bias, not useful task-state understanding.

【后来怎么看（截至冻结终点）】

Retained as the V120 performance anchor and a controlled example of capability-
preserving semantic removal.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 2654-2668。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

CI-0199, state-change ownership and full-run controls.

### CI-0201 — Log-to-source consumer mapping had higher value than extending probes

【时间 / 版本】

Post-V120 review method, 2026-08-08.

【当时在解决什么问题】

Repeated probe extensions produced diminishing returns while source and complete
logs already exposed the producer, transformation, consumer and loss semantics.

【当时的 insight / hypothesis】

Probes answer a narrow residual causal question. They should not replace the
full chain:

```text
logged metric -> exact producer -> tensor sample space -> consumer -> loss owner
-> gradient owner -> action boundary
```

【具体做法】

- Re-read G2 and G3 metric producers and discovered they measured different
  posterior spaces, invalidating a simple `.30 -> .02 means G3 collapse` story.
- Compared all V120 metrics against active source and excluded legacy zero-
  gradient groups.
- Reserved frozen interventions for remaining action-utility questions.

【观察到的结果】

Several earlier generalizations were corrected without a new probe: G/W object
identity was healthier than implied, while P1/global-K docking, P2 type routing
and duplicate bottom ingress were structurally defective.

【当时的解释】

Metric magnitude is only meaningful inside its producer's probability space.
Different diagnostics with similar names cannot be treated as a layer-to-layer
transition.

【后来怎么看（截至冻结终点）】

Retained as the preferred diagnosis order.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录
- 对话：visible messages 2662-2680。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

Source provenance, metric sample spaces and probe stopping rules.

### CI-0202 — Local P1 facts and global K dynamics needed an explicit object-chart dock

【时间 / 版本】

V120 top-down source/literature audit, 2026-08-08.

【当时在解决什么问题】

P1 read high-resolution current facts in local `[camera,8x8,M]` coordinates,
while W/P2 operated on global `K=4` objects. The existing candidate assignment
connecting these spaces was not consumed by P1.

【当时的 insight / hypothesis】

Current precision and future dynamics can only be combined coherently when they
share an explicit object-chart coordinate dock. The dock should realign local
cells/micro-values to global objects without compressing either source.

【具体做法】

- Traced `candidate_assignment [B,K,C,8,8,M]` to its unused P1 boundary.
- Compared the issue with the historical DoT idea of explicit docking and basis
  realignment, without adopting its full mechanism.
- Required P1 facts and W effects to meet on the same K/object coordinates before
  policy consequence construction.

【观察到的结果】

This was promoted to a P0 source defect and became V121's
`ObjectFactualDock`. It replaced an implicit hope that P2 could reconcile two
unrelated coordinate systems downstream.

【当时的解释】

A dock is a coordinate/identity translation, not another public latent or a
learned gain.

【后来怎么看（截至冻结终点）】

Retained as a structural principle; later P1 fidelity work revised the exact
dock representation.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收。

【证据与锚点】

- 类型：源码事实；方案/设计记录；推断/未决
- 对话：visible messages 2669-2680。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

P1 precision, global K identity and local/global coordinate bases.

### CI-0203 — Semantic, geometry and status cannot share one selection posterior by default

【时间 / 版本】

V120 P2 audit and V121 design, 2026-08-08.

【当时在解决什么问题】

P2 used a posterior derived from semantic delta to select semantic, geometry and
status values; an outer type selector then depended mostly on the action query.

【当时的 insight / hypothesis】

These fields have different evidence geometries and null semantics. Semantic
content, spatial transport/covariance and visibility/persistence should form
their own evidence-conditioned reads before a later action consumer combines
them. Shared routing lets cheap status/default values substitute for spatial
evidence.

【具体做法】

- Traced the single semantic posterior into all type values.
- Compared semantic/content, spatial moment and probability-change targets.
- Used the contemporary MV-WAM discussion only to support heterogeneous target
  geometries, not to copy its architecture.
- Planned type-specific K/V and posteriors without hard quotas or gains.

【观察到的结果】

The shared route helped explain why geometry error could improve while geometry
mass stayed near `0.114`, and why P2 could select cheap status-like value. V121
separated semantic and geometry routes; later logs exposed a distinct status
calibration bug.

【当时的解释】

Typed tensors are not typed decisions if a single posterior chooses all of them.

【后来怎么看（截至冻结终点）】

Retained; exact type inventory and algebra continued to evolve.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；方案/设计记录
- 对话：visible messages 2669-2680 and 2729。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

CI-0154, heterogeneous target geometry and P2 routing.

### CI-0204 — Repeated intent/fact injection and duplicate bottom ingress made W optional

【时间 / 版本】

V120-to-V121 top ingress audit, 2026-08-08.

【当时在解决什么问题】

S interval queries already contained goal/history/object context, but P1 injected
mean goal and latest history again. The same protected consequence also reached
bottom through both protected detail and controlled-dynamics context.

【当时的 insight / hypothesis】

Redundant same-purpose inputs are not harmless robustness. They change credit
assignment, amplify current-only information and permit the consumer to ignore
W while retaining action performance.

【具体做法】

- Traced all S-to-P1 and consequence-to-bottom paths.
- Distinguished intentional P1 current-fact access from duplicate task/history
  value injection.
- Found an old policy-layer contract could resend the post-P trajectory to the
  decoder despite the new typed-bank ingress.
- Closed the legacy midcut/layer-contract decode routes while retaining their
  training/audit data where needed.

【观察到的结果】

V121 established a single consequence ingress and reduced P3 to
precision/temporal/state-change, but subsequent evidence showed the surviving
precision/temporal definitions still contained their own redundancy/bypass.

【当时的解释】

Information conservation does not require delivering the same value twice. One
protected owner can have multiple legitimate consumers, but a terminal operation
must not receive duplicate aliases of the same fact.

【后来怎么看（截至冻结终点）】

Retained as a bottom-ingress and credit-assignment rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；历史检索
- 对话：visible messages 2676-2729。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

P1/S bypasses, single consequence ingress and policy-layer ancestry.

### CI-0205 — V121 improved object identity but inherited a hidden two-speed optimizer

【时间 / 版本】

V121 early log, 2026-08-08.

【当时在解决什么问题】

G object separation improved and P2 semantic/geometry mass balanced, but S/W
temporal differentiation and P3 precision were unexpectedly weak.

【当时的 insight / hypothesis】

Before attributing a new architecture's relative learning rates to semantics,
the optimizer must be mapped parameter by parameter. Removal of an old module
can alter a feature-test branch and silently activate legacy fallback grouping.

【具体做法】

- Traced the optimizer's branch on the existence of deleted
  `layer_contract_heads`.
- Found the fallback placed `blocks[:6]`—G1-G3, W1-W2 and P1—at `0.2x LR`.
- Found the new grounder, S, W compiler and P2/P3 at full LR.
- Matched the behavior to the logged lower learning rate.

【观察到的结果】

V121 was a two-speed top network. The error could amplify S/W/P imbalance and
made the run unsuitable as a clean long-run verdict.

【当时的解释】

Optimizer ownership is part of the architecture. Feature-presence heuristics are
unsafe when versions delete or rename modules.

【后来怎么看（截至冻结终点）】

Retained; later plans required explicit parameter-name grouping.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；V121 run confounded.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；历史检索
- 对话：visible messages 2736-2750。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`layer_contract_heads`。

【关联】

Optimizer ancestry, explicit decay/LR groups and run comparability.

### CI-0206 — Correcter teacher intervals can expose weaker student temporal bandwidth

【时间 / 版本】

V121 epoch-one comparison, 2026-08-08.

【当时在解决什么问题】

V121's W cosine was slightly better in absolute terms, yet its gap to the teacher
and variation ratio were worse.

【当时的 insight / hypothesis】

A student can look more differentiated while falling further behind a more
informative target. Absolute prediction cosine must be compared with target
cosine and normalized variation, not with a predecessor alone.

【具体做法】

- Aligned V120/V121 at batch 480 and later epoch-one windows.
- Compared object cosine separately from interval cosine.
- Compared prediction variation to target variation and aligned P2 effect/P3
  gradients.

【观察到的结果】

V121 G/W object identity improved, but S interval variation was about `0.022`
versus V120 `0.083-0.093`. W captured roughly `18-33%` of target temporal
variation, P2 effect RMS fell and P3 temporal-to-precision gradient ratio grew to
roughly `48x`. Epoch-one validation was `0.10004` versus V120 `0.09762`, with
the regression concentrated at steps 13-24.

【当时的解释】

The problem had moved downstream: objects were no longer the dominant bottleneck;
temporal intent/effect delivery was.

【后来怎么看（截至冻结终点）】

Retained as a correction to version-only cosine comparisons.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 2745-2763。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

Teacher difficulty, normalized variation and long-horizon action bands.

### CI-0207 — Zero-centered status change was misused as absolute P2 confidence

【时间 / 版本】

V121 P2 source/log audit, 2026-08-08.

【当时在解决什么问题】

P2 null mass rose to roughly `0.65-0.70` even though W had learned object
differences and visual support was valid.

【当时的 insight / hypothesis】

`visibility_change = visibility - 1` and
`persistence_change = persistence - 1` are zero-centered changes, not absolute
confidence. Feeding them into a sigmoid calibration and applying it only to
effect candidates systematically favors null even for neutral, fully valid
future.

【具体做法】

- Traced teacher status targets and P2 calibration algebra.
- Confirmed the effect prior was multiplied by calibration while null was not.
- Separated the issue from learned rejection and reliability masks.

【观察到的结果】

The formula itself pushed neutral effects below null. This explained rising
null mass more directly than “P2 chose a shortcut” alone.

【当时的解释】

Change, probability and selector support must remain different semantic types;
their numeric ranges cannot authorize substitution.

【后来怎么看（截至冻结终点）】

Retained as a confirmed V121 regression and input to V122.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2751-2773。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

CI-0197, null routing and status semantics.

### CI-0208 — Camera collapse and loss-budget imbalance starved geometry

【时间 / 版本】

V121 complete top audit, 2026-08-08.

【当时在解决什么问题】

Transport normalized error stayed near one while content dominated FutureEffect
loss; camera-specific observations were available but future geometry was weak.

【当时的 insight / hypothesis】

Two independent design defects reinforced each other: camera coordinates were
collapsed too early into `[B,K,2]`, and the future objective allocated about
`98.3%` to content while transport received about `0.5%`.

【具体做法】

- Traced per-camera object coordinates through Teacher, W and P2.
- Traced loss composition in native and normalized units.
- Distinguished missing geometry information from inadequate optimization
  pressure.
- Required camera-preserving geometry selection and per-field variance-floor
  normalization in V122.

【观察到的结果】

The source made the transport failure deterministic enough to fix without
waiting for further training. Optimizer gaps, exploding gradients, Teacher
leakage and absent P1 detail were excluded as competing explanations.

【当时的解释】

A target cannot be learned if its identity axis is removed before prediction and
its remaining error has negligible budget.

【后来怎么看（截至冻结终点）】

Retained; later schemas revisited the exact camera representation and loss
support.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2760-2773。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。

【关联】

Camera equivariance, geometry targets and loss ownership.

### CI-0209 — P3 precision duplicated the protected fact; temporal could exist without what it scheduled

【时间 / 版本】

V121 P3 audit, 2026-08-08.

【当时在解决什么问题】

P3 precision gradients collapsed while temporal dominated, even though P1 facts
and S/W innovation were available.

【当时的 insight / hypothesis】

The issue was not only weak credit: `p1_fact` passed to P3 was the exact tensor
already used as `consequence.factual_base`, so precision re-encoded the protected
fact. Meanwhile temporal could produce nonzero value from learned time identity
and action query when S/W innovation was zero.

【具体做法】

- Proved tensor identity rather than conceptual similarity for the duplicated
  fact.
- Evaluated temporal algebra under zero S/W innovation.
- Promoted both from high-confidence suspicions to confirmed source defects.
- Defined the next repair around protected bases versus innovations rather than
  “increase differentiation.”

【观察到的结果】

V122 changed precision to K-centered P1 detail and prevented learned
identity/action alone from synthesizing temporal innovation.

【当时的解释】

A lane named precision or temporal does not own that function if its value is a
duplicate fact or an unconditional adapter.

【后来怎么看（截至冻结终点）】

Retained as a terminal-consumer ownership rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 2774-2797。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`p1_fact`、`consequence.factual_base`。

【关联】

Protected facts, optional innovations and P3 lane identity.

### CI-0210 — Post-backward diagnostics are part of the training lifecycle test surface

【时间 / 版本】

V122 first start, 2026-08-08.

【当时在解决什么问题】

Preflight passed and forward/backward completed, but gradient attachment crashed
after referring to deleted P3 members.

【当时的 insight / hypothesis】

A smoke that stops at backward is incomplete when production always performs
post-backward gradient diagnostics before optimizer step. Renamed/deleted
modules must be updated in diagnostic ownership groups at the same time as the
forward graph.

【具体做法】

- Removed stale `precision_fact/precision_consequence` references.
- Registered new precision query/key/value and W object/camera groups.
- Added a full-system BF16 forward/backward test that invokes the real gradient
  diagnostic function.

【观察到的结果】

The failure occurred before optimizer step and therefore produced no valid
training update. The corrected lifecycle test passed.

【当时的解释】

Diagnostics are not passive text when they execute inside the step boundary.

【后来怎么看（截至冻结终点）】

Retained as a smoke-test completeness requirement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 2798-2801。
- 段级工件：见 Segment 022 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`precision_fact/precision_consequence`。

【关联】

Optimizer step atomicity, metric ownership and stale interface cleanup.

## Segment 022 synthesis

The controlled V120 comparison validated the completion removal and established
the strongest performance anchor so far, but source/log replay showed that the
next bottleneck was not simply “make W larger.” Precision and future dynamics
used incompatible local/global object bases; P2 shared selection across
heterogeneous semantic/geometry/status fields; current intent and consequence
entered consumers through duplicates; camera geometry was collapsed and nearly
absent from the loss budget. V121 repaired several of these boundaries and
improved object identity, but a legacy optimizer fallback created a two-speed
network, and correcter teacher intervals exposed weak student temporal bandwidth.
Further source audit found deterministic P2 status-to-null and P3 duplicate/
unconditional-lane errors, leading to V122. The segment reinforced that an
architectural repair includes optimizer and post-backward diagnostics, and that
newly informative targets can make a student look relatively worse precisely
because they reveal previously hidden capacity or routing limits.

## Segment 023 — Independent mainline extraction, failed simplification and behavior-complete recovery

### Segment boundary

```text
Date:                 2026-08-09
Visible messages:     2802-2983
Starts with:          decision to extract V122 from the historical monolith
Ends with:            server commands for mainline early-loss comparison and training
Primary historical artifacts:
  new clearvla/mainline package
  25-layer V122 launcher chain and final argv
  old trunk/runtime/config/scripts versus new typed configuration
  mainline flow, G/S/W/P, bottom, training, checkpoint and comparison tools
```

### CI-0211 — Preserve history as evidence, not as active dependency

【时间 / 版本】

V122 independent-mainline planning, 2026-08-09.

【当时在解决什么问题】

The active graph was embedded in roughly 20.8k lines of training runtime, 14.6k
lines of trunk, 13k lines of evidence code, about 90 current scripts, 415 CLI
arguments and a roughly 399-field inherited config.

【当时的 insight / hypothesis】

The execution order was still understandable; the failure was architectural
co-location and historical feature composition. A new mainline should own one
vertical configuration/model/training/runtime/checkpoint path, while old
versions remain frozen evidence rather than imported feature branches.

【具体做法】

- Built an independent `clearvla/mainline` package.
- Planned one atomic public switch after full equivalence/behavior review.
- Treated V98 and V107 only as candidate research baselines because they
  represented distinct questions, not because every intervening version deserved
  continued runtime support.
- Kept the old V122 entry frozen during extraction.

【观察到的结果】

This reduced active source closure and version branching, but the first
extraction also demonstrated that isolation alone does not guarantee faithful
migration.

【当时的解释】

Traceability belongs in Git, manifests, logs and ledgers; it should not require
every ancestor to participate in the current forward graph.

【后来怎么看（截至冻结终点）】

Retained as the correct systems goal.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测；方案/设计记录
- 对话：visible messages 2802-2817。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`clearvla/mainline`。

【关联】

CI-0171, CI-0180 and mainline source closure.

### CI-0212 — Fixed architecture choices belong in structure/manifest, not hundreds of Boolean options

【时间 / 版本】

Mainline configuration extraction, 2026-08-09.

【当时在解决什么问题】

The first idea risked turning 399 old fields into a smaller JSON without removing
their invalid combinatorics.

【当时的 insight / hypothesis】

Capability-defining facts such as 3-2-3 topology, four intervals, K=4, one P1
read and single bottom ingress are not independent tunable flags. They should be
encoded in constructors and ABI identity; only genuine dimensions, numerical
parameters and training hyperparameters remain configurable.

【具体做法】

- Created a compact manifest/config.
- Removed dead/ghost switches not consumed by the new graph.
- Treated any later behavior change as an ABI/schema change rather than a hidden
  Boolean combination.

【观察到的结果】

The config became more locally explainable. Later audits still found ghost
fields and stale ABI numbers, showing that consumption tests were required in
addition to schema validation.

【当时的解释】

A configuration surface is a claim about executable variability. An unused knob
is a false experiment contract.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 2818-2824 and 2849-2850。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Capability manifests, dead configuration and experiment identity.

### CI-0213 — Online evidence and training targets should be different types, not one dictionary plus permission flags

【时间 / 版本】

Mainline typed-boundary extraction, 2026-08-09.

【当时在解决什么问题】

Online RGB/raw evidence, future DINO supports, full future state/action targets
and deployment inputs lived in one dictionary guarded by
`allow_future_training_evidence`.

【当时的 insight / hypothesis】

Teacher isolation is stronger when future fields cannot exist in the deployment
input type. Numeric validation should occur at construction, not repeatedly in
the five-step hot path.

【具体做法】

- Split online observation, training future supports and action/state targets
  into non-interchangeable types.
- Kept Teacher targets FP32/no-grad and reachable only from training APIs.
- Created a smaller deployment cache distinct from training build context.

【观察到的结果】

Replacing future supports changed Teacher targets while online W remained
bit-exact. Later cache review removed dense charts, candidate assignments and
reconstruction targets that had no ODE-loop consumer.

【当时的解释】

Function signatures and data types can enforce a lifecycle boundary that Boolean
runtime checks only describe.

【后来怎么看（截至冻结终点）】

Retained as a mainline design invariant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 2820-2824 and 2847-2848。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`allow_future_training_evidence`。

【关联】

Teacher isolation, minimal deployment cache and typed lifecycles.

### CI-0214 — Observation grounding must not inherit the action rollout carrier

【时间 / 版本】

G extraction during mainline refactor, 2026-08-09.

【当时在解决什么问题】

Old G queries came from a generic rollout carrier containing multiple historical
W/owner/action-era fields. Copying the entire state would reproduce the monolith
inside a new directory.

【当时的 insight / hypothesis】

G establishes current facts and must be observation-only. Language enters S,
future enters Teacher and noisy action enters P/bottom; those restrictions should
be visible in signatures and gradient paths.

【具体做法】

- Defined compact G working state containing only observation bank, G1 alignment,
  G2 rectification and G3 typed facts.
- Kept M, camera, chart and typed owner axes.
- Moved P1 to one static high-resolution factual read.
- Removed retired W/horizon/diagnostic fields from G state.

【观察到的结果】

The extracted G/P1 path ran forward/backward, but later review found the first
version had omitted active host blocks and therefore was not behavior complete.

【当时的解释】

Semantic ownership and parameter/capacity preservation are separate obligations:
a cleaner signature does not authorize replacing an active computation.

【后来怎么看（截至冻结终点）】

Retained after restoring the active G/P1 hosts.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留原则；first extraction corrected.

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 2825-2829。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Observation-only G and mandatory subsystem familiarity.

### CI-0215 — Equal axis sizes can hide a semantic transposition from every shape test

【时间 / 版本】

Mainline recognizer audit, 2026-08-09.

【当时在解决什么问题】

`interval_identity` missed a singleton dimension. Since both interval count and
object count were four, all shapes remained valid while interval identity was
attached to the object axis.

【当时的 insight / hypothesis】

Shape correctness cannot validate axis identity when two dimensions have equal
extent. Permutation/Jacobian fixtures must label and independently vary each
axis.

【具体做法】

- Corrected the missing axis insertion.
- Added semantic axis tests rather than only expected-shape assertions.
- Audited other `mean/expand/detach/normalize` uses for collapsed-and-recreated
  axes.

【观察到的结果】

The bug offered a concrete explanation for some interval/K publicization despite
apparently valid contracts.

【当时的解释】

Axis semantics are data, not metadata inferred from dimension length.

【后来怎么看（截至冻结终点）】

Retained as a canonical axis-contract failure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible message 2841。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`interval_identity`、`mean/expand/detach/normalize`。

【关联】

K permutation, interval permutation and silent broadcasting.

### CI-0216 — A masked target must retain the hidden fact

【时间 / 版本】

Mainline Pre-G/JEPA audit, 2026-08-09.

【当时在解决什么问题】

The mask removed cells from the online chart and then used the same masked tensor
as the reconstruction target, making the target zero exactly where recovery was
intended.

【当时的 insight / hypothesis】

Context corruption and target construction must be separate. Otherwise a masked
reconstruction objective rewards predicting the mask token/zero rather than the
held-out fact.

【具体做法】

- Preserved a frozen full DINO target before applying the online mask.
- Rebuilt G reconstruction against that full target.
- Kept the mask out of the target path and deployment unaffected.

【观察到的结果】

The objective gained an actual completion pressure without adding a new loss
weight.

【当时的解释】

An objective's name cannot compensate for a target that has already discarded
the answer.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 2842-2846。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Masked modeling, target isolation and observable evidence.

### CI-0217 — A predictor must not propose and verify its own address with the same transport

【时间 / 版本】

Mainline W/P2 audit, 2026-08-09.

【当时在解决什么问题】

P2 generated an address from W transport and then compared that address with the
same transport as evidence, creating a self-confirming coordinate score.

【当时的 insight / hypothesis】

Proposal and verification need independent information sources. The action query
may propose a destination; W's supervised transport/address may score it. Using
one field for both creates high confidence without new evidence.

【具体做法】

- Made P2 destination proposals action-query owned.
- Kept W transport/address as the supervised factual value/score source.
- Implemented real probability-chart translation rather than reweighting near
  the source position.
- Required neutral W to yield exact-zero optional P3 lanes.

【观察到的结果】

The P2 loop was removed while the same FutureDynamics field remained the only W
consumer input.

【当时的解释】

Cycle consistency is useful only when the compared quantities have independent
provenance.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 2842-2846。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Address ownership, transport supervision and P2 coordinate scoring.

### CI-0218 — Runtime counters, caches and validators must measure real calls and real consumers

【时间 / 版本】

Mainline lifecycle/performance audit, 2026-08-09.

【当时在解决什么问题】

Static/Teacher build counters were hard-coded; validation rebuilt
RGB-to-G-to-S-to-W-to-P1 twice; deployment cache retained large training-only
state; each ODE step risked repeated numerical validation.

【当时的 insight / hypothesis】

Lifecycle contracts must be instrumented at actual module boundaries. A constant
counter is a self-assertion, and a cache is only minimal if every retained field
has a hot-path consumer.

【具体做法】

- Replaced constants with true call counters.
- Shared one typed static cache between validation loss and five-step sampling.
- Split training build context from minimal deployment cache.
- Moved static numeric reductions out of per-step validators.
- Deferred Teacher diagnostics and GPU-to-CPU metric materialization to logging
  boundaries.

【观察到的结果】

The changes removed redundant static builds, unused chart/target lifetimes and
many synchronization kernels without changing network values.

【当时的解释】

Lifecycle health is observable behavior, not a comment or metric constant.

【后来怎么看（截至冻结终点）】

Retained as runtime review policy.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 2847-2855 and 2874。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Static/dynamic frequency, hot-path synchronization and cache ownership.

### CI-0219 — The first independent bottom rewrite was an unauthorized capability reduction

【时间 / 版本】

Early mainline bottom extraction, 2026-08-09.

【当时在解决什么问题】

To escape the old runtime, the first extraction substituted a simplified bottom
before fully mapping the active Evidence MMDiT, action proposal, transition and
execution boundaries.

【当时的 insight / hypothesis】

This was not a safe refactor. A subsystem cannot be simplified from names,
manifests or parameter impressions; every active producer, block, consumer,
loss and deployment call must be mapped first.

【具体做法】

- Withdrew the simplified bottom after static review.
- Audited actual active deterministic Evidence MMDiT/execution behavior rather
  than relying on historical `CVAE/workspace_v1` naming.
- Preserved three action blocks, typed selection/value, low-rank capacity and
  differentiable execution depth.
- Removed only source-proven duplicate current-state injection, type-as-value and
  full block-by-candidate replay.

【观察到的结果】

The later parameter audit revealed that the same first extraction had also
omitted active top hosts and transition/proposal modules. The user's challenge
changed the acceptance standard from “cleaner graph” to a proven one-to-one
active-algorithm inventory.

【当时的解释】

Cleanliness is not fidelity. Removing a tangled implementation before
understanding its active behavior is architecture replacement, not refactoring.

【后来怎么看（截至冻结终点）】

Retained as a major failure and review-method correction. Historical bottom
naming statements are not current architecture authority.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已否定 first rewrite；lesson retained.

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录
- 对话：visible messages 2834-2840, 2861-2864 and 2914-2927。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`CVAE/workspace_v1`。

【关联】

Mandatory subsystem familiarity, information loss and user trust.

### CI-0220 — Low-rank capacity and continuation were nominal but algebraically bypassed

【时间 / 版本】

Mainline bottom audit, 2026-08-09.

【当时在解决什么问题】

`effective_basis_mass≈29` was interpreted as a low-rank update, and three
continuation stages were interpreted as soft dwell.

【当时的 insight / hypothesis】

The algebra contradicted both names. Disabling a 32-dimensional projection while
passing the remaining roughly 480 dimensions left a high-dimensional bypass;
independent continuation gates allowed a later layer to continue after an
earlier layer stopped.

【具体做法】

- Replaced the capacity path with a true ordered rank-32 projection where zero
  capacity gives exact zero and full capacity stays in the low-rank subspace.
- Replaced independent continuation probabilities with cumulative local stay
  products, enforcing non-increasing survival with depth.
- Kept execution cost detached/audit-only.

【观察到的结果】

The controller semantics became consistent with the metrics and received direct
action gradients without replaying full blocks for every candidate.

【当时的解释】

A bottleneck is defined by the residual subspace actually reaching the consumer,
not the size of one gated branch.

【后来怎么看（截至冻结终点）】

Retained as a bottom algebra correction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 2860-2868 and 2946-2949。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Hidden high-dimensional bypasses, soft dwell and execution ownership.

### CI-0221 — Flow required one shared spatial axis, displacement unit and temporal reference

【时间 / 版本】

Mainline flow audit, 2026-08-09.

【当时在解决什么问题】

Flow loss, G addresses, Teacher transport and action utility had long failed to
align. The extraction review found multiple mutually reinforcing coordinate
errors.

【当时的 insight / hypothesis】

The learned field was a boundary-scaled displacement owned by the source grid,
but consumers treated it as normalized displacement on the current grid.
Additionally, raw pair `-4 -> 0` represented four frames while Teacher scaled it
as though one field covered 48 frames.

【具体做法】

- Anchored current facts at current chart coordinates instead of re-warping them
  by previous-to-current flow.
- Defined flow as real previous-to-current displacement on the current chart.
- Unified warp/cycle/Teacher units and recorded reference frame span `4`.
- Scaled future offsets relative to four frames rather than 48.
- Kept flow as a soft prior, with semantic matching able to override errors.

【观察到的结果】

The repair removed a structural reward for zero flow and aligned spatial and
temporal units without imposing nonzero-flow quotas.

【当时的解释】

A geometry field is identified by direction, owning grid, coordinate units and
time interval. Matching only tensor shape leaves all four ambiguous.

【后来怎么看（截至冻结终点）】

Retained as the canonical flow contract.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 2853-2856 and 2902-2907。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Flow-DINO, current chart anchoring and horizon units.

### CI-0222 — A learnable feature-only warp objective can erase time instead of learning motion

【时间 / 版本】

Mainline numerical flow review, 2026-08-09.

【当时在解决什么问题】

Warp loss compared features from a shared learnable raw encoder at both times,
leaving a zero-flow/temporally invariant representation shortcut.

【当时的 insight / hypothesis】

If both sides of a correspondence objective can adapt, they can reduce loss by
removing temporal sensitivity. A fixed observable anchor is needed within the
same flow objective, but it must not force motion where RGB is static.

【具体做法】

- Kept feature warp.
- Added a low-resolution RGB photometric anchor under the existing external flow
  weight.
- Vectorized the radius-two neighborhood grid sampling from roughly 156 small
  kernels across directions/refinements to batched local sampling.

【观察到的结果】

The objective gained an external observable reference while preserving evidence
resolution; performance improved without adding a flow quota or new deployment
solve.

【当时的解释】

Anti-collapse anchors should constrain correspondence evidence, not prescribe a
target flow magnitude.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 2907-2913。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Zero-flow shortcuts, observable anchors and kernel factorization.

### CI-0223 — Checkpoint and output-directory operations must be validated before mutating live state

【时间 / 版本】

Mainline runtime/checkpoint audit, 2026-08-09.

【当时在解决什么问题】

Partial bottom migration, bad optimizer shapes, dtype drift, scheduler mismatch
or an occupied run directory could contaminate a live model/run before an error
was raised.

【当时的 insight / hypothesis】

Resume and migration are transactions: validate full identity, keys, shapes,
dtypes, finite values, optimizer/scheduler states and destination run identity
before applying any mutation.

【具体做法】

- Made bottom migration all-or-nothing.
- Validated Adam state shapes and scheduler base LR before model load.
- Added dtype/finite checks and a unique dtype resolver across train/validation/
  sampling.
- Defined fresh output as empty and resume output as empty or identity-matching.
- Included semantic data config and decoded/DINO cache identities while ignoring
  only relocatable path strings.
- Separated train-flow, DataLoader, preflight and validation RNG states and saved
  them in checkpoints.

【观察到的结果】

Bad restores were rejected before live state changed, and resume could no longer
silently switch BF16/FP32 or append metrics from another experiment.

【当时的解释】

Recovery correctness is state-transition atomicity, not just matching parameter
names.

【后来怎么看（截至冻结终点）】

Retained as a mainline runtime invariant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 2847, 2857, 2875-2888 and 2891-2892。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Exact resume, RNG replay and experiment directories.

### CI-0224 — The first refactor accidentally removed roughly 49.3M active parameters

【时间 / 版本】

Mainline parameter reconciliation, 2026-08-09.

【当时在解决什么问题】

The first independent graph had only about `117.14M` total/trainable parameters,
far below old V122's `227.47M total / 166.36M trainable`.

【当时的 insight / hypothesis】

This was not benign removal of frozen legacy. The trainable difference was
almost exactly explained by active computations omitted during extraction.
The user's demand to stop saying “possibly” established a stronger rule: every
active old algorithm needs an explicit equivalent before deletion is allowed.

【具体做法】

- Replayed the final 25-layer launcher and actual source call graph.
- Recovered three active G host blocks and one P1 host (`~31.51M`).
- Recovered the eight-row action-history proposal (`~10.01M`).
- Recovered controlled dynamics/transition (`~7.89M`).
- Restored training-time exact-null condition dropout for goal `5%`, action
  history `10%` and proposal `25%` with owned RNG.
- Kept only source-proven frozen/skipped blocks and legacy heads removed.

【观察到的结果】

The active-parameter gap of roughly `49.3M` closed. The corrected graph reached
about `169.38M` trainable-scale parameters before later action/T5 fixes; its
trainable count slightly exceeded old V122 while total count remained lower due
to removed frozen/skipped modules.

【当时的解释】

Parameter accounting was used as a conservation audit, not a capacity target.
The first extraction was an unapproved architecture reduction, not a faithful
refactor.

【后来怎么看（截至冻结终点）】

Retained as the most consequential refactor failure in this segment. It directly
motivates the mandatory subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已修复；lesson retained.

【证据与锚点】

- 类型：源码事实；过程/决策记录
- 对话：visible messages 2914-2949。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`227.47M total / 166.36M trainable`。

【关联】

Active-module inventories, information conservation and interaction trust.

### CI-0225 — Final launcher resolution, not ancestor defaults, defines the physical action algorithm

【时间 / 版本】

Mainline physical-action recovery, 2026-08-09.

【当时在解决什么问题】

The first refactor replaced the old action representation with direct 7D flow and
omitted fixed training algorithms inherited through the script stack.

【当时的 insight / hypothesis】

The real baseline was the final resolved launch configuration: an internal 18D
physical action field (`6 arm absolute + 6 arm delta + 6 gripper field`) decoded
to native 7D, with anchor-band weighting, consistency/event/motion objectives and
information-balanced sampling. Ancestor defaults later overridden to zero must
not be resurrected.

【具体做法】

- Captured final argv from the full launcher chain.
- Restored 18D source/noisy query/velocity/ODE while keeping clean history,
  proposal and deployment output native 7D.
- Restored `decode_delta_blend=0.25`, `(4,12,24)` anchor bands, raw-unit gripper
  event, focal event, motion and decoded/delta consistency.
- Restored the resolved `0.50/0.125/0.375` information-balanced training sampler.
- Explicitly did not restore ancestor auxiliary weights overridden to zero.

【观察到的结果】

The physical-field widening added exactly `11,275` parameters. The main action
codec/noise fixtures matched exactly, and execution capacity/continuation
received nonzero main-action gradients.

【当时的解释】

Script ancestry is a program; the final evaluated configuration, together with
the active source branch, is the behavior contract.

【后来怎么看（截至冻结终点）】

Retained as the V120/V122 recovery methodology.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；历史检索
- 对话：visible messages 2938-2949。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.50/0.125/0.375`。

【关联】

Physical action geometry, launch provenance and loss equivalence.

### CI-0226 — Input ABI omissions can be hidden until a real artifact is loaded

【时间 / 版本】

Mainline T5/dtype production-size check, 2026-08-09.

【当时在解决什么问题】

The new config declared `goal_token_dim=768`, but the production T5 artifact was
`[39,4096]`; DINO cache was FP16 while training autocast used BF16, causing mixed
low-precision spatial sampling failure.

【当时的 insight / hypothesis】

Synthetic tests and type hints do not establish a production input ABI. Real
serialized artifacts and dtype combinations must be loaded before declaring a
refactor complete.

【具体做法】

- Corrected T5 projection to `4096 -> 512` and advanced manifest identity.
- Standardized spatial sampling in local FP32 and cast results back to caller
  dtype.
- Ran a full-size B1 CUDA BF16 forward/backward with 336 RGB, 576 DINO patches,
  12 future supports and real T5.

【观察到的结果】

Parameter count rose by exactly `1,703,936` for the correct language projection,
to about `171.09M`. Full B1 used roughly `2.674 GiB` allocated and completed in
about 3.56 seconds on the local GPU.

【当时的解释】

Artifact-backed smoke tests close dimensions and dtype domains that reduced
fixtures cannot reveal.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 2953-2972。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Language ABI, FP16/BF16 sampling and production-size tests.

### CI-0227 — Refactor comparison should lock common semantics, not demand identical loss

【时间 / 版本】

V122-versus-mainline comparison-tool design, 2026-08-09.

【当时在解决什么问题】

The user wanted a same-data early comparison on an 8 GB GPU. The two graphs had
different parameter order/topology and intended structural fixes, so exact loss
equality was impossible and undesirable.

【当时的 insight / hypothesis】

Lock what should be shared—sample order, normalized/native action targets, flow
time/noise, batch and optimizer hyperparameters—and compare common loss scale,
composition, gradients and short-run slopes. Version-specific G/S/W/P auxiliary
losses should be reported separately.

【具体做法】

- Built sequential-process comparison to avoid simultaneous GPU residency.
- Verified old/new action codec encode/decode and same-seed source noise were
  exact.
- Captured old final argv rather than hand-recreating it.
- Classified metrics as close, same order or different order with causal notes.

【观察到的结果】

The comparison tool was delivered for server runs; local data lacked the real
HDF5 action/state source, so no local fixture result was represented as a formal
dataset comparison.

【当时的解释】

Equivalence testing must distinguish intentionally changed internals from shared
behavioral invariants.

【后来怎么看（截至冻结终点）】

Retained as migration validation methodology. Subsequent independent-mainline
and V120 recovery audits used the same locked-common-semantics principle; the
original one-off server comparison was superseded by fuller source/log replay.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续恢复审计吸收；方法保留。

【证据与锚点】

- 类型：源码事实；日志观测；数据探针；过程/决策记录；推断/未决
- 对话：visible messages 2953-2983。
- 段级工件：见 Segment 023 boundary 的 `Primary historical artifacts`。

【关联】

Golden fixtures, early training slopes and controlled comparison.

## Segment 023 synthesis

The independent-mainline effort exposed both the value and danger of a large
refactor. Typed online/teacher boundaries, compact capability configuration,
minimal static caches and direct source closure removed real systems debt. The
same review found deep inherited faults—axis aliasing, masked-zero targets,
self-confirming address scores, fake lifecycle counters, flow coordinate/time
unit errors, learnable feature collapse and nominal low-rank/continuation
bypasses. But the first “clean” extraction also crossed the line into unapproved
architecture reduction, removing roughly 49.3M active G/P1/proposal/transition
parameters and the 18D physical action algorithm. User challenge forced the
correct recovery standard: resolve the final launcher, map every active module,
loss and consumer, and permit deletion only with source proof of frozen, skipped,
diagnostic or duplicate behavior. The corrected mainline therefore became not
the smallest graph, but the smallest graph that could account for all active
behavior and every deliberate semantic repair.

## Segment 024 — V122 failure replay and first independent-mainline experimental falsification

### Segment boundary

```text
Date:                 2026-08-10
Visible messages:     2984-3047
Starts with:          complete V122 eight-epoch audit
Ends with:            source review begins for mainline behavior lost relative to V120
Primary historical artifacts:
  complete v122.log
  v120_long.log
  first independent mainline long log
  schema-16 mainline G/S/W/action changes and production configuration
```

### CI-0228 — V122 was an optimization-success, policy-failure version

【时间 / 版本】

Completed V122, 2026-08-10.

【当时在解决什么问题】

V122 had received extensive structural repairs and its training losses decreased,
but its full eight-epoch policy quality had not yet been judged against V120.

【当时的 insight / hypothesis】

A run that optimizes its training and representation objectives can still be a
failed policy when pressure concentrates on near horizons, hold behavior and
public latent directions.

【具体做法】

- Audited all eight epochs of action, horizon bands, gripper/event, G/S/W/P,
  flow and execution metrics.
- Compared to the controlled V120 long run rather than only V121 or an early
  V122 prefix.
- Mapped each metric to the frozen V122 source before checking inheritance in the
  new mainline.

【观察到的结果】

- Best validation RMSE was about `0.08914`, `12.4%` worse than V120's
  `0.07931`; epoch eight rebounded to `0.09109`.
- Steps 1-4 improved roughly `34%`, but steps 13-24 only `4.6%`; tail/first grew
  to about `9.24`.
- Gripper event recall fell from about `0.383` to `0.191` and F1 from `0.374`
  to `0.289`, while motion F1 stayed healthy.
- G slot cosine rose `0.493 -> 0.710` and overlap `0.282 -> 0.575`.
- W losses improved, but prediction interval variation ended at `0.082` versus
  teacher `0.139` and stayed more public.

【当时的解释】

V122 learned internal geometry/future objectives but failed to convert them into
long-horizon action utility; near action and hold shortcuts absorbed credit.

【后来怎么看（截至冻结终点）】

Retained as a failed structural evidence version, not a performance parent.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已否定为性能基线；保留作故障证据。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 2985-3004。
- 段级工件：见 Segment 024 boundary 的 `Primary historical artifacts`。

【关联】

Training-validation decoupling, horizon imbalance and gripper conservatism.

### CI-0229 — A clean refactor that faithfully inherits a failed optimizer geometry can reproduce the failure

【时间 / 版本】

Pre-run schema-16 mainline risk audit, 2026-08-10.

【当时在解决什么问题】

The independent mainline had removed runtime debt but still inherited some V122
W/action objective algebra.

【当时的 insight / hypothesis】

Code cleanliness and behavior fidelity are insufficient if the behavior anchor
itself is a failed version. The refactor needed to retain V120's effective
capability while correcting V122's condition competition and horizon/event
credit allocation.

【具体做法】

- Identified W's intent and coarse-action branches as being added before a shared
  saturating nonlinearity, allowing the larger branch to erase the other.
- Identified row-count-weighted horizon loss and dominant hold rows in gripper.
- Preserved total action budget while redistributing across semantic horizon
  bands and actual event rows.
- Required no-event batches to remain algebraically bit-equivalent.

【观察到的结果】

The changes passed local algebra/tests, but the ensuing long run still failed
the V120 action and structure anchor. This showed the identified risks were real
but not the complete cause.

【当时的解释】

Repairing an optimization bias cannot compensate for an upstream representation
objective that has a symmetric public solution.

【后来怎么看（截至冻结终点）】

Partially retained; exact V120 recovery required further source reconstruction.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留，insufficient alone.

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 2993-3019。
- 段级工件：见 Segment 024 boundary 的 `Primary historical artifacts`。

【关联】

W condition competition, horizon weighting and gripper event emphasis.

### CI-0230 — Per-slot reconstruction can still have a symmetric public solution

【时间 / 版本】

Schema-16 G repair and first long run, 2026-08-10.

【当时在解决什么问题】

To stop V122 slots from jointly explaining one target, the mainline changed G
prototype reconstruction so each slot reconstructed its own posterior read.

【当时的 insight / hypothesis】

The modification removed one mixture shortcut but did not establish different
ownership: if every slot chooses the same evidence, every per-slot target is also
the same. Conditional reconstruction alone therefore retains a fully symmetric
public optimum.

【具体做法】

- Compared the new target algebra with V120's object behavior.
- Tracked G pair cosine from initialization through the eight-epoch mainline run.
- Aligned W object-effect cosine downstream.

【观察到的结果】

Mainline G and W object cosines remained approximately `1.000` through training,
where V120 epoch-eight values were roughly `0.483` and `0.440`. The new target
did not create global-K identity.

【当时的解释】

An owner-conditioned target only differentiates owners if the conditioning
assignment is already informative or jointly constrained.

【后来怎么看（截至冻结终点）】

Retained as the primary explanation for the first mainline's object collapse.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；schema-16 G objective superseded.

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3010-3019 and 3033-3047。
- 段级工件：见 Segment 024 boundary 的 `Primary historical artifacts`。

【关联】

CI-0194, symmetric slots and reconstruction ownership.

### CI-0231 — Image size does not determine cached token-grid size

【时间 / 版本】

First mainline server smoke, 2026-08-10.

【当时在解决什么问题】

The new config assumed 336-pixel input implied a 24x24/576-token DINO cache,
while the actual production cache contained 16x16/256 tokens per camera.

【当时的 insight / hypothesis】

Preprocessing geometry is an artifact ABI and must be read from cache metadata.
Image resolution alone does not reveal backbone patch stride, crop or serialized
token count.

【具体做法】

- Verified old and new launchers used the same cache path.
- Restored config/default to 256 tokens.
- Kept strict actual-versus-expected cache validation.
- Verified adaptive pooling from 16x16 to the model's 8x8 chart.

【观察到的结果】

The smoke proceeded without rebuilding the cache; 88 mainline tests passed.

【当时的解释】

Production cache metadata, not a guessed backbone geometry, defines the input
contract.

【后来怎么看（截至冻结终点）】

Retained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3020-3029。
- 段级工件：见 Segment 024 boundary 的 `Primary historical artifacts`。

【关联】

Artifact ABIs and real-data smoke tests.

### CI-0232 — The correct performance anchor changed the conclusion from “matched” to “failed”

【时间 / 版本】

First independent-mainline eight-epoch result, 2026-08-10.

【当时在解决什么问题】

The first read compared mainline mainly to failed V122 and described it as having
roughly matched the predecessor. The user corrected the intended baseline to
`v120_long.log`.

【当时的 insight / hypothesis】

Baseline selection is part of causal analysis. A failed immediate parent is
useful for detecting new regressions but cannot define acceptable performance.

【具体做法】

- Verified `v120_long.log` was the complete binder-fixed, completion-off V120,
  not the pre-fix or partial run.
- Aligned all eight validation points, horizons and structural metrics.

【观察到的结果】

Mainline was worse at every epoch: about `0.10933 vs 0.09762` at epoch one,
`0.09834 vs 0.08312` at epoch four, best `0.09107 vs 0.07931`, and epoch eight
`0.09127 vs 0.08145`. It never reached V120's epoch-four result. G/W publicization
was also dramatically worse.

【当时的解释】

The mainline was not merely slower. It had replaced effective V120 behavior with
a symmetric G/W solution while preserving an easy action path.

【后来怎么看（截至冻结终点）】

Retained. V120 became the primary behavior/performance recovery anchor for the
subsequent schemas.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 3030-3047。
- 段级工件：见 Segment 024 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`v120_long.log`。

【关联】

Performance versus regression anchors and experiment interpretation.

## Segment 024 synthesis

Complete V122 evidence showed that a technically stable, internally improving
version could still fail as a policy through near-horizon, hold and public-latent
shortcuts. The independent mainline then demonstrated a second-order danger: a
clean refactor can remove systems debt and even repair some V122 biases while
still replacing the last healthy V120 object behavior with a symmetric
reconstruction optimum. Using V122 as the comparison briefly hid the failure;
restoring V120 long as the behavior anchor made it unambiguous at every epoch.
The next task therefore became exact source/behavior reconstruction, not another
local loss adjustment or a claim that all named modules had been migrated.

## Segment 025 — Behavior-level V120 recovery audit and the first repair closure

### Segment boundary

```text
Date:                 2026-08-11
Visible messages:     3048-3140
Starts with:          first systematic independent-mainline replay against exact V120 source/log
Ends with:            Schema18/19 repair closure and the boundary between local tests and recovery
Primary historical artifacts:
  exact V120 commit and long-run log
  independent-mainline source and complete log
  Schema18/19 source, tests and recovery diagnostics
```

These historical repairs are not current implementation authority; several were
revised by later schemas.

### CI-0233 — Migration review must compare behavior, not module names or parameter totals

【时间 / 版本】

Independent mainline to V120 replay, 2026-08-11.

【当时在解决什么问题】

The independent mainline contained modules named G, S, W, P and bottom, yet its
eight-epoch behavior was consistently inferior to V120. A module inventory or
parameter count could no longer explain whether useful V120 behavior survived.

【当时的 insight / hypothesis】

A refactor can preserve a name while changing the operator, target, ownership or
consumer semantics. The audit therefore had to classify every dataflow segment
as an active omission, a non-equivalent replacement, a reasonable replacement,
or historical code that had never been active.

【具体做法】

- Anchored V120 to exact commit `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`.
- Walked observation/Pre-G → G → S → Teacher/W → P1/P2/P3 → transition →
  bottom/execution → losses/runtime.
- Compared source, serialized run identity and all eight epochs rather than
  inferring behavior from current legacy files.
- Kept initial causal claims explicitly falsifiable until source and log evidence
  agreed.

【观察到的结果】

The mainline had fewer total parameters but more trainable parameters than V120;
much of the reduction came from removing inactive frozen ancestry. The damaging
differences were concentrated semantic/operator changes, not the wholesale loss
of an active CVAE or workspace.

【当时的解释】

“Cleaner” source structure was not evidence of behavioral fidelity. The unit of
recovery had to be an end-to-end contract with a real consumer and gradient owner.

【后来怎么看（截至冻结终点）】

Retained as a durable audit method. Later failures repeatedly confirmed that
named interfaces and passing shape tests can coexist with lost information.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 3049-3060。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`0b92d359a2889a0a1b1eba256007c00ccbc54f3c`。

【关联】

V120 behavior anchor, source provenance and the later subsystem-familiarity rule.

### CI-0234 — The first reconstructed causal chain was multiplicative, not one missing block

【时间 / 版本】

Independent mainline long-run diagnosis, 2026-08-11.

【当时在解决什么问题】

Determine why a stable model that still learned from `0.1093` to `0.0913` never
matched V120 and plateaued above it at every epoch.

【当时的 insight / hypothesis】

Several individually survivable changes compounded: homogeneous G objects made
Teacher/W/P1 object-indistinguishable; P3 multiplicative gates further suppressed
the weak signal; a rank-32 bottom operator then discarded most remaining update
directions; weaker far-horizon weighting reduced the pressure to recover them.

【具体做法】

Mapped source equations to log evidence, including G/W object cosine, S interval
variation, P1 detail activity, P3 gradients, transition support and bottom
capacity semantics.

【观察到的结果】

- G and W object cosine stayed approximately `1.0`.
- V120 G identity gradually separated and its epoch-eight mean object cosine was
  reported near `0.483`.
- Current S interval variation was about `0.0418` versus V120 about `0.137`.
- The simple action/state/history route remained trainable, explaining why the
  model improved without recovering object/future behavior.

【当时的解释】

The failure was a continuous attenuation chain, not a binary deletion. Each
boundary made the optional structured path easier to ignore until the bottom
could solve a simpler stateless/near-horizon task.

【后来怎么看（截至冻结终点）】

Partly retained. The exact implementation details changed later, but the insight
that several legal attenuations can compose into a functional bypass remained
central through Schema39.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；具体修法被后续方案多次改写。

【证据与锚点】

- 类型：日志观测；方案/设计记录；推断/未决
- 对话：visible messages 3054-3060 and 3065-3068。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

G identifiability, P1 local facts, P3 gating, bottom bandwidth and far-horizon loss.

### CI-0235 — Inactive V120 ancestry was explicitly excluded from recovery

【时间 / 版本】

V120 source replay, 2026-08-11.

【当时在解决什么问题】

The large V120 parameter total suggested that the independent mainline might
have accidentally removed a CVAE, hierarchical workspace or execution system.

【当时的 insight / hypothesis】

Serialized names and checkpoint paths are not proof that a component was active.
Only the executed forward, optimizer ownership and run flags can identify the
behavioral baseline.

【具体做法】

- Verified V120's active bottom was a deterministic organizer plus three Evidence
  MMDiT layers; the variational CVAE and hierarchical workspace were disabled.
- Verified `fresh=1` and `stage1_initialization_enabled=0` despite a serialized
  historical Stage1 checkpoint path.
- Verified the old contraction depth controller was frozen and that two logged
  operator gradient labels referred to the same active basis, not two controllers.

【观察到的结果】

V120 had about `235.66M` total and `166.61M` trainable parameters, versus about
`171.36M` total and `171.25M` trainable in the then-current mainline. The total
drop therefore did not prove that an active main path had disappeared.

【当时的解释】

Mechanical restoration of disabled ancestry would add cost and ambiguity without
recovering behavior. The active equations, not the old object graph, were the
proper recovery target.

【后来怎么看（截至冻结终点）】

Retained. This was an important corrective to the earlier fear that the whole
bottom CVAE/workspace had been lost.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；历史检索；推断/未决
- 对话：visible messages 3060, 3090 and 3117-3120。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Run-context provenance and the ban on restoring dead historical modules.

### CI-0236 — `capacity=1` had been changed from full-width identity into a rank-32 ceiling

【时间 / 版本】

Schema17/18 bottom-capacity audit, 2026-08-11.

【当时在解决什么问题】

Capacity gradients disappeared and the reconstructed bottom plateaued despite
otherwise finite training.

【当时的 insight / hypothesis】

The current operator used `Q diag(m) Q^T u`, so even full capacity retained only
the learned rank-32 subspace. V120 instead used
`u - Q diag(1-m) Q^T u`; full capacity was exact 512-dimensional identity and
the basis described only removable directions.

【具体做法】

- Restored exact-zero, exact-full-identity and non-expansive capacity semantics.
- Kept FP32 QR for the basis.
- Added 0/1/intermediate capacity tests and checked its real optimizer owner.

【观察到的结果】

For approximately isotropic updates, the bad full-capacity equation could retain
only about `sqrt(32/512)=0.25` of RMS. It also began with capacity near `0.906`
and continuation multipliers near `0.9/0.81/0.729`, unlike V120's initial full
capacity/full dwell.

【当时的解释】

This was an independent, deterministic bandwidth regression: a control intended
to remove selected directions had become a maximum-rank bottleneck.

【后来怎么看（截至冻结终点）】

Retained as a confirmed historical defect and a canonical example of a same-name,
different-algebra migration error.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在当时修复；原则保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3055-3056, 3060 and 3069。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`sqrt(32/512)=0.25`。
- 直接量值/表达式（卡内原文）：`0.9/0.81/0.729`。

【关联】

Bottom information bandwidth, optimizer decay and later complaints about
unnecessary compression.

### CI-0237 — Horizon weighting had silently shifted training mass from tail to first steps

【时间 / 版本】

Action objective replay, 2026-08-11.

【当时在解决什么问题】

The reconstructed mainline showed weak distant actions and a near-horizon
shortcut even after structural modules were restored.

【当时的 insight / hypothesis】

The new code normalized each horizon band by its length before combining bands.
That was not equivalent to V120's per-row `1.0/1.1/1.2` weighting.

【具体做法】

Computed the effective row-mass distribution and restored V120's per-row
far-emphasis while leaving the newer gripper event-balanced weighting as a
separate mechanism.

【观察到的结果】

V120 allocated about `14.9% / 32.3% / 52.8%` to rows 1-4, 5-12 and 13-24. The
new normalization changed this to about `31.3% / 32.8% / 35.8%`, roughly doubling
near pressure and cutting far pressure by about one third.

【当时的解释】

This objective drift compounded weak S/W signals. It was not legitimate to
attribute all distant failure to architecture while the loss had also changed.

【后来怎么看（截至冻结终点）】

Retained as a confirmed confounder. Later schemas kept action-loss changes
separate from top-representation changes.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在当时修复。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3060 and 3069。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1.0/1.1/1.2`、`14.9% / 32.3% / 52.8%`、`31.3% / 32.8% / 35.8%`。

【关联】

Tail RMSE, long-horizon S/W pressure and controlled experiment identity.

### CI-0238 — Restoring visual history required motion alignment, not fixed-cell differencing

【时间 / 版本】

Schema18 observation/Pre-G repair, 2026-08-11.

【当时在解决什么问题】

The independent mainline retained three robot-state history points but only the
current DINO frame and one raw image pair, deleting a V120 visual/flow segment.

【当时的 insight / hypothesis】

The missing `(-8,-4,0)` visual history and two adjacent flow intervals were real
information loss, but simply subtracting the same chart cell across frames would
confound object motion with semantic change.

【具体做法】

- Restored three visual history frames and two learned-flow segments.
- Warped `-8` and `-4` evidence toward current coordinates before forming history
  innovation.
- Kept online history separate from Teacher future evidence and cached it once per
  observation.

【观察到的结果】

Focused tests showed history changed online owner evidence without modifying the
training target, and deployment did not rebuild it at each Euler node.

【当时的解释】

Information restoration must preserve geometry and causal provenance; otherwise
an apparently richer input can introduce a new false correspondence.

【后来怎么看（截至冻结终点）】

Partly retained. The need for aligned history survived; exact later Flow-DINO/
JEPA designs evolved.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留，被后续视觉历史设计吸收。

【证据与锚点】

- 类型：测试/运行观测；推断/未决
- 对话：visible messages 3060 and 3071-3075。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Flow-DINO, history ownership and static observation caching.

### CI-0239 — G had both a symmetric target and active blocks whose output was not consumed

【时间 / 版本】

Schema18 G repair, 2026-08-11.

【当时在解决什么问题】

Grounding reconstruction improved from about `1.43` to `0.76` while G object
cosine remained approximately `1.0` for the whole run.

【当时的 insight / hypothesis】

Two defects coexisted. A shared public carrier was injected identically into all
local candidates, and a self-referential prototype loss used each slot's own soft
read to create the content it was asked to reconstruct. Moreover, the three G
role blocks only updated reconstruction-side `public_scene_base`, not the online
binding key consumed by policy.

【具体做法】

- Removed deterministic common-value injection and the movable self-target.
- Returned dense spatial reconstruction as the principal grounding objective.
- Routed role-block output into competitive address keys rather than copying a
  common value into every object.
- Preserved camera axes, typed posteriors, flow geometry and G3 recomputation.

【观察到的结果】

Source inspection explained how four uniform slots could all predict a global
mean and lower reconstruction without acquiring identity. “The blocks execute”
was therefore not proof that their result reached the action path.

【当时的解释】

An objective can be nontrivial numerically yet admit a completely symmetric
solution; and an active trainable block can still be functionally dead at its
intended consumer.

【后来怎么看（截至冻结终点）】

Retained as diagnosis, but later G1-G3 fidelity work showed this repair was not
yet a complete recreation of V120 grounding.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；当时修复不完整，后来继续重做。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 3058, 3060, 3070 and 3085-3089。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`public_scene_base`。

【关联】

Global-K identifiability, G1-G3 ownership and self-reconstruction shortcuts.

### CI-0240 — P1 needed protected local-chart support independent of a healthy global K

【时间 / 版本】

Schema18 P1 repair, 2026-08-11.

【当时在解决什么问题】

The current P1 cropped only around global-K camera coordinates and centered
object context across K. When all K slots were equal, keys, coordinates and all
microgrid reads also became equal.

【当时的 insight / hypothesis】

V120's useful escape path was the retained local `[camera,8,8,M]` address lattice:
global K docked object identity, but was not the sole source of high-resolution
facts. P1 should remain able to read local evidence while K is still learning.

【具体做法】

- Introduced query-specific reads over the full local chart before the packed
  3×3 high-resolution detail read.
- Removed a newly discovered detach on G assignment so action→P1 could exert
  ordinary ownership pressure on G.
- Added query/coordinate intervention and gradient tests.

【观察到的结果】

Queries and coordinates changed independently, and action-side P1 gradients could
reach G assignment in focused tests. V120 reference metrics had shown active
micro values near `0.083`, spatial variation near `0.041` and late-detail update
near `1.314`.

【当时的解释】

A global object abstraction must not become an irreversible precision bottleneck.
Protected local facts are complementary support, not a bypass to be deleted in
order to force K learning.

【后来怎么看（截至冻结终点）】

Retained as a core principle. Later replay found the mechanical V120 P1 extraction
still differed from this first reconstruction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：原则保留；具体实现后来被替换。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 3060 and 3070-3075。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Static P1, FactualPrecisionDock/FactualIntentDock and information conservation.

### CI-0241 — P2/P3 anti-bypass constraints had become a serial multiplicative failure

【时间 / 版本】

Schema18/19 P2/P3 review, 2026-08-11.

【当时在解决什么问题】

P2 contained well-bounded semantic/geometry selection and exact-zero future
effects, yet P3 precision and temporal gradients were nearly absent.

【当时的 insight / hypothesis】

P3 centered detail across homogeneous K and then multiplied precision, temporal
and state-change by weak consequence/action gates. This prevented a bypass in
name but allowed one weak W signal to erase all factual and temporal lanes.
Fixed interval identity and noisy-action conditioning also offered alternate
routes around observable S/W innovation.

【具体做法】

- Kept protected factual precision available.
- Required only future-specific correction to depend on W.
- Replaced serial multiplicative dependence with additive, bias-free typed
  consequence interaction.
- Removed fixed-interval and noisy-action temporal shortcuts from the relevant
  decisions.

【观察到的结果】

Focused tests established exact neutral-W semantics and ordinary gradients, but
no production run had yet demonstrated V120 recovery.

【当时的解释】

Preventing a shortcut must not destroy the information that the intended route
needs to operate. A dependency is healthy only when the owner can add a meaningful
innovation, not when it can veto every downstream source.

【后来怎么看（截至冻结终点）】

Retained as an important principle; later schemas repeatedly revisited the exact
P2/P3 decomposition and showed the first repair was not final.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；具体代数后来多次修改。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 3060 and 3089-3095。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

FutureEffect ownership, optional innovation and later P2/P3 lane separation.

### CI-0242 — A difference of two learned networks is not an exact-zero counterfactual

【时间 / 版本】

Schema18 controlled-transition review, 2026-08-11.

【当时在解决什么问题】

The transition was described as `coeff(real)-coeff(neutral)`, but the real and
neutral branches used different learned queries, memory lengths, directional
biases and stochastic dropout.

【当时的 insight / hypothesis】

Subtracting two unmatched functions does not create a centered action effect.
Even zero proposal can leave a learnable constant “consequence” carrier.

【具体做法】

- Used the same network, memory shape and query geometry for factual and
  zero-proposal counterfactual evaluation.
- Avoided independent dropout noise in the subtraction.
- Recovered dense `4×2×8×8=512` spatial directions, then query-pooled them to
  four action-basis tokens per horizon (`96` total) instead of one token per
  horizon or 512 persistent bottom tokens.

【观察到的结果】

Zero proposal became exact zero by construction, and the transition retained
more spatial/action-basis identity without the full attention cost.

【当时的解释】

Counterfactual zero semantics require matched computation, not merely subtractive
naming. Compression should occur only after the consumer-relevant basis exists.

【后来怎么看（截至冻结终点）】

Retained as a durable causal and numerical lesson. The particular transition
representation later evolved.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：原则保留；具体载体被后续实现吸收。

【证据与锚点】

- 类型：因果干预
- 对话：visible messages 3060, 3071, 3079 and 3082。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Matched-noise ablations, neutral consequence and delayed compression.

### CI-0243 — Low-reliability future targets must first become valid neutral targets

【时间 / 版本】

Schema19 Teacher/S/W target repair, 2026-08-11.

【当时在解决什么问题】

Association reliability could be low because the posterior was diffuse, not only
because null was correct. Removing reliability weighting outright would amplify
a spatially averaged ambiguous future; retaining it as a mask left W an
unsupervised writable region.

【当时的 insight / hypothesis】

Target semantics must be repaired before changing loss weights. Low-confidence
association should continuously fall back to current content, zero transport and
current unit-mass address, and that neutral target should remain supervised.

【具体做法】

- Added continuous Teacher fallback for content, transport and address.
- Removed the second reliability attenuation from S/object losses after the
  target itself became neutral.
- Bumped the training ABI because checkpoint targets had changed.

【观察到的结果】

Focused tests showed ambiguous rows had an explicit neutral target rather than
either a noisy averaged future or an unmasked free carrier.

【当时的解释】

Masking uncertainty is not equivalent to defining what “no reliable change”
means. A weakly supervised latent can become a shortcut even if its output is
called future dynamics.

【后来怎么看（截至冻结终点）】

Partly retained. Later exact-V120 Teacher replay rejected some entropy/reliability
shrinkage and restored a different target algebra, but the general lesson about
not creating free latent space survived.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；Schema22/23 later revised the exact algebra.

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 3093 and 3095-3098。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Teacher exact target, reliability diagnostics and future loss support.

### CI-0244 — Decoded action events and dedicated event/motion heads had been logged under the same names

【时间 / 版本】

Validation-semantics audit, 2026-08-11.

【当时在解决什么问题】

Gripper and motion F1 appeared comparable to V120, but the current validator used
normalized decoded-action changes and a `0.05` threshold while V120 used raw-unit
event labels and dedicated endpoint head logits.

【当时的 insight / hypothesis】

Identical metric names do not imply identical labels or predictions. This was an
evaluation ABI error, not direct evidence of model regression.

【具体做法】

- Returned final five-step event/motion logits from the sampler.
- Split decoded-action events from dedicated head metrics with explicit names.
- Added tests preventing either source from impersonating the other.

【观察到的结果】

The earlier same-name F1 values were declared non-comparable to V120. The repair
restored the ability to distinguish action-derived events from endpoint heads.

【当时的解释】

Metric provenance is part of model identity. An audit can otherwise invent a
behavioral story from a logging mismatch.

【后来怎么看（截至冻结终点）】

Retained. A later Schema23 replay refined endpoint timing further, but this
producer/label distinction remained necessary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在当时修复；后续继续细化。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3076-3078。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Endpoint-head time contract and validation ABI.

### CI-0245 — Compact console logging and lossless archival are separate products

【时间 / 版本】

Schema19 observability repair, 2026-08-11.

【当时在解决什么问题】

The same zero-value filter was used for nohup output and `metrics.jsonl`. Exact
collapse of W variation, P2 effect, flow or a head would disappear from the
archive as if the metric had never been connected. Several compact logger keys
also did not match actual engine-produced names.

【当时的 insight / hypothesis】

Human-readable output may omit ordinary zeros; the machine archive must never do
so. A metric configured for display but never produced is worse than a missing
metric because it creates false confidence.

【具体做法】

- Preserved every active scalar, including exact zeros, in JSONL.
- Kept console output compact while retaining explicit contract-zero values.
- Added actual W interval metrics, owner gradients, action bands and source-aware
  names.
- Taught the audit tool to ingest `metrics.jsonl` and `run_context.json` and to
  verify seed, data, normalizer and optimizer identity.

【观察到的结果】

A BF16 training step archived all `543` active scalars including `88` exact zeros,
while the console projected `462`. The recovery tool correctly rejected the old
mainline with exit code 3 and separated 15 failures/missing-evidence items.

【当时的解释】

Without lossless telemetry, collapse becomes indistinguishable from missing
wiring, and a later audit cannot reconstruct causality.

【后来怎么看（截至冻结终点）】

Retained and repeatedly reinforced by later large-log analyses.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；历史检索
- 对话：visible messages 3076-3078, 3092, 3100 and 3108-3140。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`metrics.jsonl`、`run_context.json`。

【关联】

Recovery audit, run identity and gradient-failure telemetry.

### CI-0246 — Optimizer geometry was a hidden part of the V120 behavior contract

【时间 / 版本】

Optimizer/source replay, 2026-08-11.

【当时在解决什么问题】

The independent mainline assigned a uniform LR multiplier and weight decay where
V120 used role-specific update geometry.

【当时的 insight / hypothesis】

Even with identical forward equations, changing the relative learning speed of
history/proposal, decoder and contraction bases can favor the easy bottom path
over G/S/W. Optimizer ownership and decay therefore belong to experiment identity.

【具体做法】

- Recovered the observed V120 proposal LR (`5e-5`, about `0.625×` main), decoder
  multiplier (`0.7×`) and contraction-basis multiplier (`2×` decoder, no decay)
  where the active operator semantics matched.
- Logged every role's LR and ownership.
- Explicitly did not restore frozen depth controls or infer benefit from the old
  Stage1 path.

【观察到的结果】

The source difference was consistent with a faster bottom/history shortcut and
weaker top owners, though the exact causal performance effect still required a
new long run.

【当时的解释】

This was treated as recovery of known training geometry rather than free
hyperparameter tuning, while non-equivalent obsolete multipliers were excluded.

【后来怎么看（截至冻结终点）】

Retained as an experiment-identity principle. Later V120 optimizer-decay replay
made the name-based no-decay rules more precise.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：原则保留；具体分组后来继续校正。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 3099 and 3113-3121。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Schema24 V120 decay recovery and optimizer ownership.

### CI-0247 — Matched interventions were added because nonzero gradients do not prove action utility

【时间 / 版本】

Schema19 validation diagnostics, 2026-08-11.

【当时在解决什么问题】

Hundreds of internal metrics and nonzero gradients still could not answer whether
proposal or execution updates helped or harmed decoded actions.

【当时的 insight / hypothesis】

Action-level, matched-noise interventions are needed to establish functional
closure. A path may be trainable and statistically active while degrading the
policy.

【具体做法】

- Added proposal-zero, bottom-no-updates and bottom-full-updates validation on a
  small configured batch subset.
- Reused the same cache and initial noise.
- Scoped proposal-zero to clean proposal entering P1/transition, leaving genuine
  action history intact; scoped execution interventions to bottom update strength.
- Added a small end-to-end validation test covering teacher forcing, five-step
  sampling and all matched ablations under autocast/cache aggregation.

【观察到的结果】

The infrastructure could identify “zero is better” as a harmful path rather than
mistaking its gradient for evidence of value. Local tests reached 121 passing,
but the assistant correctly kept production CUDA and eight-epoch recovery open.

【当时的解释】

Functional closure requires a measurable change at the downstream consumer, not
only a Jacobian or loss owner inside the producer.

【后来怎么看（截至冻结终点）】

Retained. Later schemas expanded this into matched S/W/P and lane-level
interventions.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预
- 对话：visible messages 3122-3134。
- 段级工件：见 Segment 025 boundary 的 `Primary historical artifacts`。

【关联】

Causal ablations, matched noise and the later consumer-to-producer review rule.

## Segment 025 synthesis

The first behavior-level replay established that the failed independent mainline
was neither a simple parameter deletion nor a single broken block. Several
same-name operators had changed algebra, local visual support had been made
dependent on an already homogeneous global K, and anti-shortcut gates had become
information-destroying serial dependencies. At the same time, the audit rejected
two tempting but false restorations: inactive V120 CVAE/workspace ancestry and a
frozen operator-depth controller. Schema18/19 attempted a coherent repair across
G, P1, transition, P3, capacity, targets, optimizer geometry and observability.
The most durable outcome was methodological: source provenance, lossless zero
logging, matched interventions and consumer-visible closure became requirements.
Passing 121 local tests was explicitly not accepted as V120 recovery; production
BF16 behavior and the full eight-epoch gate remained open.

## Segment 026 — Schema19 falsification and reconstruction of the refactor accident

### Segment boundary

```text
Date:                 2026-08-11
Visible messages:     3141-3240
Starts with:          recovery-gate and experiment-identity hardening
Ends with:            recognition that the low-memory mainline redesigned mature P/transition/bottom
Primary historical artifacts:
  Schema19 source and run through batch 1580/epoch one
  V120 codec, P/transition/bottom source
  normalizer fingerprints and recovery audit
```

### CI-0248 — Comparable experiment identity requires compatible algorithms, not string prefixes

【时间 / 版本】

Schema19 recovery-gate completion, 2026-08-11.

【当时在解决什么问题】

The new gate compared V120's 12-character MD5 of six-decimal normalizer statistics
with a full SHA-256 from the current run, and could therefore reject identical
normalizers or accept an invalid prefix convention.

【当时的 insight / hypothesis】

A fingerprint is meaningful only together with its serialization and hashing
algorithm. Cross-version comparability should store a legacy-compatible identity
for equality and a full modern digest for archival disambiguation.

【具体做法】

- Reproduced V120's short normalizer fingerprint exactly as `b9a2b34d6697`.
- Stored both the compatible short MD5 and full SHA-256 with distinct semantics.
- Verified split, sampler and normalizer source rather than only matching paths.
- Corrected optimizer run-context fields to distinguish base LR, group initial
  LR, multiplier, tensor count and element count.

【观察到的结果】

The recovery gate could compare like with like, while the full digest remained
available to detect ambiguity. This also exposed that tool/CLI execution needed
testing separately from library unit tests.

【当时的解释】

Experiment identity is an algorithmic contract, not a label. A broken comparison
tool can manufacture either a false regression or a false recovery.

【后来怎么看（截至冻结终点）】

Retained as a provenance requirement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 3141-3150 and 3157-3163。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`b9a2b34d6697`。

【关联】

Run context, normalizer identity and recovery gates.

### CI-0249 — Raw objective values and weighted contributions must both be logged

【时间 / 版本】

Schema19 loss-ledger audit, 2026-08-11.

【当时在解决什么问题】

The new action objective boosted gripper-event rows but retained the name
`action_flow`, while the recovery gate compared it to V120's unboosted physical
flow. The JSONL also lacked each term's actual weighted contribution.

【当时的 insight / hypothesis】

An objective may be reasonable yet incomparable to a baseline metric. Logging
only raw auxiliary losses cannot establish which term actually dominates
backward.

【具体做法】

- Logged the true optimized event-balanced objective separately from an unbalanced
  `*_v120_comparable` value.
- Added each term's exact weighted contribution and a loss-closure residual.
- Classified all 287 active V120 metrics against the 544/568 current metrics by
  semantics rather than count or name.

【观察到的结果】

The earlier current-mainline physical-flow values could no longer be treated as
directly comparable merely because of a shared label. The audit retained one
real mechanism gap—execution-value calibration—without mechanically restoring
every old metric producer.

【当时的解释】

Loss telemetry needs both the physical error and the optimized objective. Otherwise
a deliberate weighting change masquerades as a model-quality regression.

【后来怎么看（截至冻结终点）】

Retained. Later plans restored V120 action semantics before reintroducing event
boost under an explicit contract.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 3154-3171。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`action_flow`。

【关联】

Event-positive boost, loss closure and cross-version RMSE comparability.

### CI-0250 — First-step zero gradients at a zero-initialized boundary were not a permanent dead path

【时间 / 版本】

Schema19 optimizer-ownership audit, 2026-08-11.

【当时在解决什么问题】

A synthetic first training step showed zero gradients in several upstream W/P2,
consequence and organizer parameters, suggesting another nominal-but-dead path.

【当时的 insight / hypothesis】

A zero-initialized downstream projection can intentionally delay upstream
gradients for one update. One snapshot cannot distinguish staged activation from
a permanent disconnection.

【具体做法】

Ran five consecutive optimization steps and tracked every trainable tensor under
21 owners; proposed a regression requiring all expected owners to wake by the
second real update.

【观察到的结果】

All trainable tensors acquired nonzero gradients from step two onward. The first
step was an initialization-order effect, not a lasting bypass.

【当时的解释】

Gradient liveness must be interpreted over the initialization schedule. The
opposite error—accepting a healthy aggregate while internal owners remain dead—
also required owner-level tracking.

【后来怎么看（截至冻结终点）】

Retained as a caution against both false alarms and aggregate-gradient complacency.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；方案/设计记录
- 对话：visible messages 3166-3169。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Zero initialization, owner diagnostics and gradient-spike analysis.

### CI-0251 — The action codec was suspected, compared, and exonerated

【时间 / 版本】

V120/Schema19 action-geometry replay, 2026-08-11.

【当时在解决什么问题】

An identical 18-D shape could have hidden a switch from V120's arm/gripper
residual geometry to a different manifold/Parseval representation.

【当时的 insight / hypothesis】

Shape equality is insufficient; source/noise/decode algebra must be compared.

【具体做法】

Verified V120's active configuration was `legacy_independent +
legacy_handcrafted(6)`, with independent Gaussian sources and the same `0.25`
delta decoding used by the current mainline.

【观察到的结果】

The coordinate chart, noise source, decoder and base physical-flow equation were
equivalent. Upstream information and optimization/consumption remained the
credible causes.

【当时的解释】

The investigation prevented a speculative codec rewrite from being added to an
already broad repair.

【后来怎么看（截至冻结终点）】

Retained as a successfully falsified hypothesis.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已否定为本轮根因。

【证据与锚点】

- 类型：源码事实；推断/未决
- 对话：visible messages 3172-3178。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Physical versus normalized action metrics and source-algebra equivalence.

### CI-0252 — Schema19 improved G/W but left action behavior almost identical to the failed mainline

【时间 / 版本】

Schema19 batch 20-1580, 2026-08-11.

【当时在解决什么问题】

Assess whether the large source repair changed the policy before committing to
eight epochs.

【当时的 insight / hypothesis】

If upstream structure changes but matched-batch action curves remain almost
identical, the new information is not functionally consumed by the main task.

【具体做法】

Aligned 79 training points among Schema19, the old independent mainline, V120 and
V122 using the unboosted comparable physical-flow metric; inspected G/S/W/P,
transition and execution metrics at the same batches.

【观察到的结果】

- Schema19 versus old-mainline physical-flow correlation: `0.99970`.
- Total-loss correlation: `0.99975`.
- G pair cosine improved from about `0.996` to `0.484`, chart overlap to `0.165`.
- W pair cosine improved from `1.0` to `0.338`, better separated than matched V120.
- At batch 1580, Schema19 physical flow was about `0.1532` versus V120 `0.0888`.

【当时的解释】

The optimizer continued to use a public temporal/global transition/full-bottom
route even though G/W now produced more discriminative internal facts. This was
not evidence that G/W needed more capacity.

【后来怎么看（截至冻结终点）】

Retained. It became the clearest early example that healthier producer metrics
do not prove downstream use.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 3196-3204。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Functional closure, matched-batch curves and action-path bypasses.

### CI-0253 — The shared K+null chain destroyed the fact before it became “protected”

【时间 / 版本】

Schema19 P1→P2→P3 diagnosis, 2026-08-11.

【当时在解决什么问题】

P1 high-resolution detail remained substantial, but consequence and P3 precision
collapsed toward zero.

【当时的 insight / hypothesis】

P1 aggregated only K mass from a K+null posterior without renormalizing after
discarding null. P2 inherited the same posterior/prior and P3 precision used it
again. A single learned null therefore rejected current fact, future effect and
precision together.

【具体做法】

Traced the posterior through all three consumers and reconstructed factual-base
RMS from logged effect/interaction/consequence quantities.

【观察到的结果】

- P1 detail remained about `0.324 RMS`.
- The factual base fell from roughly `0.34-0.44` at batch 20 to about `0.0056`
  at batch 1580, at least a 62× reduction.
- P3 precision reached about `0.000615` versus V120 about `0.260`.
- Semantic/geometry null mass was about `0.748/0.782` at batch 1580 and near
  `0.9994` in epoch-one validation.

【当时的解释】

The base was protected only after a destructive optional router. V120 selected
addresses but placed high-resolution current fact outside optional null rejection.

【后来怎么看（截至冻结终点）】

Retained as a confirmed information-loss defect and a major precursor to later
protected-base/null rules.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；具体修复进入后续 integrated mainline。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 3202-3209 and 3231-3236。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Protected factual base, learned null and optional innovation.

### CI-0254 — S learned a growing common carrier while temporal differentiation vanished

【时间 / 版本】

Schema19 S diagnosis, 2026-08-11.

【当时在解决什么问题】

S interval variation remained close to initialization and temporal variation
was almost always omitted as exact zero, even as S/history/action RMS grew.

【当时的 insight / hypothesis】

All intervals read the same global goal/history values; 24 temporal queries then
read four nearly identical interval values. P3 accepted this common temporal
source additively without null. S could therefore become a high-amplitude public
adapter without representing time.

【具体做法】

Compared interval and temporal variation with object/history/action innovation
and P3 temporal-base RMS across 79 windows and against V120.

【观察到的结果】

Interval variation stayed near `0.0155`; temporal variation appeared only six
times, never above about `3.6e-5`, and was zero after batch 480. Meanwhile history
innovation grew roughly `0.331→0.536`, action innovation `0.521→0.626`, and P3
temporal base `0.335→0.583`. V120 at batch 1580 had interval/temporal variation
near `0.105/0.048`.

【当时的解释】

Large RMS was not temporal capacity. It was a common conditional carrier whose
query labels gave the appearance of a time axis.

【后来怎么看（截至冻结终点）】

Retained as an early statement of the recurring S publicization problem.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；后续多版继续处理。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3201-3209 and 3231-3236。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

S interval identity, typed innovation and public/typed separation.

### CI-0255 — Normalizing cosine logits and then dividing by `sqrt(512)` forced transition pooling to uniformity

【时间 / 版本】

Schema19 controlled-transition diagnosis, 2026-08-11.

【当时在解决什么问题】

The repaired transition built 512 distinct spatial rows, but its 96 output tokens
appeared to carry almost the same information.

【当时的 insight / hypothesis】

L2-normalized query/key dot products already lie in `[-1,1]`; dividing again by
`sqrt(512)` limits logits to about `±0.044`. A 512-way softmax is then necessarily
near uniform.

【具体做法】

Mapped the exact equation to `controlled_transition_pool_entropy` and separated
pre-pooling spatial variation from post-pooling query variation.

【观察到的结果】

Entropy remained `0.999999-1.000000`. The dense rows existed, but the 96 pooled
values approximated the same global average and provided a cheap action/history
carrier.

【当时的解释】

This was a deterministic scale error, not insufficient training. Logging variation
before the destructive consumer had falsely suggested that space survived.

【后来怎么看（截至冻结终点）】

Retained as a confirmed implementation defect and a textbook example of checking
metrics at the wrong side of a bottleneck.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；当时方案后续废弃。

【证据与锚点】

- 类型：源码事实；日志观测；方案/设计记录；推断/未决
- 对话：visible messages 3201-3212 and 3231-3236。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`controlled_transition_pool_entropy`。

【关联】

Delayed compression, consumer-side diagnostics and transition bandwidth.

### CI-0256 — BF16 sigmoid saturation turned adaptive execution into a fixed full-depth shortcut

【时间 / 版本】

Schema19 bottom-controller diagnosis, 2026-08-11.

【当时在解决什么问题】

Capacity and continuation quickly became exactly one and their gradients vanished.

【当时的 insight / hypothesis】

Without candidate-value calibration, action loss rewards opening all available
blocks. A high-LR BF16 sigmoid reached representable `1.0`, after which capacity
and continuation gradients were exactly zero.

【具体做法】

Reconstructed the timeline and compared it with V120's execution reader and
controller gradients.

【观察到的结果】

Capacity rose from about `0.906` at batch 20 to `0.996` at batch 160 and exact
`1.0` by batch 180; expected depth became exact `3.0`, all 32 basis directions
were active and controller gradients stayed zero. V120 still had value/controller
gradients near `0.061` at batch 1580.

【当时的解释】

The controller no longer adapted; it granted the bottom maximum capacity to fit
the public shortcut. It was not proven to cause P collapse, but it removed a
constraint that might have forced structured evidence use.

【后来怎么看（截至冻结终点）】

Retained as a confirmed saturation failure. Later work debated how much of V120's
candidate replay/value machinery should be restored.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；原 Schema19 controller 被判不合格。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3202-3212 and 3231-3236。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Execution value, BF16 saturation and gradient spikes.

### CI-0257 — Execution-value supervision was active and useful, but not a universal cure

【时间 / 版本】

V120/V122 execution replay, 2026-08-11.

【当时在解决什么问题】

The independent mainline had removed V120's `0.05` candidate-value objective,
retaining only action-driven capacity/continue control. It was unclear whether
the expensive path was active capability or historical overhead.

【当时的 insight / hypothesis】

V120's reader learned real relative candidate value, but V122 showed that a
healthy value reader alone could not prevent top-level publicization and policy
regression.

【具体做法】

Checked weighted contribution, gradient, correlation, pair accuracy and top-1
accuracy across V120 and V122, and distinguished value supervision from
audit-only execution cost.

【观察到的结果】

At V120 batch 1580 the contribution was about `0.0207`, gradient `0.0614`, value
correlation `0.52` and pair accuracy `0.76`; later correlation/pair/top-1 reached
roughly `0.845/0.94/0.855`. V122 could retain healthy ranking yet still fail.

【当时的解释】

The path was not dead and its deletion was not behavior-equivalent, but restoring
it as a single patch would not repair G/S/W/P.

【后来怎么看（截至冻结终点）】

Retained with this qualification. Later replay distinguished the mature bottom
behavior worth preserving from controller complexity that still required audit.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；有效机制但非充分条件。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 3156-3158, 3166 and 3208-3212。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.845/0.94/0.855`。

【关联】

Candidate replay, value calibration and resource cost.

### CI-0258 — A surprisingly low 9.93 GB peak exposed that the “refactor” had deleted active computation

【时间 / 版本】

Schema19 epoch-one resource/architecture review, 2026-08-11.

【当时在解决什么问题】

The run used about 9.93 GB instead of V120's roughly 16 GB. Initially the higher
throughput was attributed to caching, vectorized logging and removed ancestry.

【当时的 insight / hypothesis】

Parameter memory could not explain the multi-gigabyte reduction: the current
model had about 5.2M more trainable parameters, while removed frozen BF16 weights
accounted for only about 0.13 GB. The missing memory therefore came largely from
activation-producing computation.

【具体做法】

Replayed V120 and current P1, transition and bottom source; compared token/candidate
counts, execution replay and activation-producing branches.

【观察到的结果】

- Mature V120 P1's four glimpses, 49 fine candidates and protected detail had
  been replaced by early K+null aggregation.
- Dense transition was reduced 512→96 through near-uniform pooling.
- Up to seven differentiable execution candidates and the value reader had been
  replaced by one-pass sigmoid controls.
- Reasonable savings from caching/Teacher isolation existed but could not explain
  the whole drop.

【当时的解释】

Low memory was not intrinsically bad; here it was partly the other face of lost
information bandwidth and missing differentiable execution work.

【后来怎么看（截至冻结终点）】

Retained. It directly motivated later insistence that resource improvements be
explained by module and activation differences, not celebrated in isolation.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；历史检索
- 对话：visible messages 3183-3195 and 3210-3214。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

Memory accounting, P1 precision bandwidth and execution replay.

### CI-0259 — The independent-mainline creation commit was a redesign accident, not an extraction

【时间 / 版本】

Git ancestry reconstruction, 2026-08-11.

【当时在解决什么问题】

The user challenged why already mature P and bottom code had been reimplemented
and whether prior instructions to preserve the lower path had been ignored.

【当时的 insight / hypothesis】

The error began in commit `91a4237` (`feat: add independent ClearVLA mainline`),
which added about 18,758 lines and newly implemented observation, G/S/W/Teacher,
P, transition, bottom and runtime. Schema19 commit `51f18ad` then extended that
new design instead of correcting its foundation.

【具体做法】

Located the exact commits and inventoried the first version of
`ObjectFactualReader`, `ObjectPolicyPlanCompiler`,
`ControlledTransitionDynamics` and `EvidenceMMDiTBottom`, including the explicit
decision to omit candidate replay.

【观察到的结果】

The mainline was not a reorganized V120 graph. It preserved class names and broad
roles while replacing active address bandwidth, execution training and numerical
semantics. Earlier README claims that active P/bottom behavior had not been
deleted were retracted.

【当时的解释】

Old source organization had been wrongly equated with old algorithmic behavior.
“Understand and rewrite” was substituted for “extract active behavior and repair
proven defects,” contrary to the user's stated boundary.

【后来怎么看（截至冻结终点）】

Retained as a major implementation-process failure and a direct ancestor of the
mandatory subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；错误归因已确认。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 3213-3221。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`91a4237`、`51f18ad`。

【关联】

Commit `91a4237`, Schema19 and behavior-preserving refactoring.

### CI-0260 — “Audit-style refactor” replaced both blind rollback and free redesign

【时间 / 版本】

Post-incident planning, 2026-08-11.

【当时在解决什么问题】

The user did not want either an uncritical V120 rollback or another architecture
invented during cleanup; the goal was to identify and repair unreasonable parts
without losing stable behavior.

【当时的 insight / hypothesis】

The correct rule became:

- mechanically preserve effective stable forward/optimization behavior;
- locally repair defects supported by source and logs;
- remove only demonstrably inactive/version-only code;
- retain uncertain behavior until it is understood.

【具体做法】

The proposed integrated foundation restored V120's active visual/P1/dense
transition/Evidence bottom/execution-value/action geometry, then selectively
admitted newer mechanisms with evidence: Teacher isolation, source-relative flow,
typed object/camera axes, bounded G ownership and exact neutral counterfactuals.
The failed S, shared-null chain, near-uniform pooling and one-pass bottom were
excluded.

【观察到的结果】

Schema19 epoch one confirmed the mixed diagnosis: G/W identity improved, but
RMSE was `0.10689` versus V120 `0.09762`; P1 spatial variation, S temporal,
P2 effect, P3 precision and execution control were severely degraded.

【当时的解释】

The project did not need to discard all post-V120 work. It needed V120 active
behavior as the mechanical substrate and newer improvements as explicit,
individually justified adapters.

【后来怎么看（截至冻结终点）】

Retained as the intended recovery method, though later implementation still fell
short repeatedly and prompted even stricter end-to-end familiarity requirements.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；后续执行质量需由后续段落评估。

【证据与锚点】

- 类型：日志观测；过程/决策记录；方案/设计记录
- 对话：visible messages 3214-3240。
- 段级工件：见 Segment 026 boundary 的 `Primary historical artifacts`。

【关联】

V120-grounded integrated mainline, provenance fixtures and information
conservation.

## Segment 026 synthesis

Schema19 delivered a decisive negative result despite local tests and real G/W
improvement. The current and failed-mainline action curves were nearly identical,
because protected P1 facts were attenuated by a shared null, S produced a growing
common temporal carrier, transition attention was mathematically forced toward a
uniform average, and BF16 execution controls saturated to an all-open dead state.
The unexpectedly low memory footprint then exposed the deeper historical cause:
the independent-mainline creation commit had reimplemented mature P, transition
and bottom behavior instead of extracting it. The conversation explicitly
corrected the project method from free redesign to audit-style refactoring—keep
stable active behavior, repair only proven defects, preserve uncertain mechanisms
and measure functional consumption at the action endpoint.

## Segment 027 — Mechanical V120 extraction and adapter-boundary reconstruction

### Segment boundary

```text
Date:                 2026-08-11
Visible messages:     3241-3369
Starts with:          byte-verified extraction of the local V120 source core
Ends with:            adapter-boundary errors found before the first Schema20 run
Primary historical artifacts:
  .audit/v120_exact_source_0b92d359/
  vendored V120 import closure and serialized profile
  Schema20 visual/P1/transition/Evidence-bottom adapters
```

“Restored module” remained a source claim, not yet a behavioral result.

### CI-0261 — A trusted historical baseline had to be materialized and verified before adaptation

【时间 / 版本】

V120-core extraction for Schema20, 2026-08-11.

【当时在解决什么问题】

Repeated `git show` snippets and command-output truncation had already encouraged
partial reconstructions and missed file tails.

【当时的 insight / hypothesis】

Behavior recovery needs a complete, locally inspectable source oracle plus its
serialized runtime profile. Neither memory nor a current legacy copy is adequate.

【具体做法】

- Extracted the exact V120 commit in fixed-size blob chunks.
- Detected and discarded a first copy contaminated by terminal truncation markers.
- Replaced line counting that omitted blank lines with record/byte verification.
- Verified 33 initial files byte-for-byte modulo line endings and materialized 96
  non-default serialized model fields.
- Reduced the vendored core to 19 files actually reachable from the active entry
  rather than retaining trunk/system/legacy copies.

【观察到的结果】

The core imported independently and provided a stable source oracle. Later
checkpoint identity used the real import closure instead of every `.py` under a
directory.

【当时的解释】

A recovery source can itself be corrupted or incomplete; import success alone
does not prove its tail or profile is intact.

【后来怎么看（截至冻结终点）】

Retained. The complete `.audit/v120_exact_source_0b92d359/` snapshot later became
the preferred ancestry source.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；历史检索
- 对话：visible messages 3243-3253 and 3316-3318。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`.audit/v120_exact_source_0b92d359/`。

【关联】

V120 exact source snapshot and source provenance fixtures.

### CI-0262 — Schema20 restored mature active modules instead of matching their names

【时间 / 版本】

Schema20 integrated-mainline implementation, 2026-08-11.

【当时在解决什么问题】

Schema19's custom ConvGRU observation, `ObjectFactualReader`, pooled transition
and one-pass bottom were not V120 behavioral extractions.

【当时的 insight / hypothesis】

The active V120 Flow-DINO/address bank, LateRawDetail P1, 512-row transition and
Evidence MMDiT/execution graph had to become the actual instantiated classes. New
G/S/W/P logic could only enter through an explicit seam.

【具体做法】

- Replaced the simplified transition with 512 spatial rows and a centered
  proposal delta.
- Replaced the custom bottom with extracted
  `EvidenceLatentMMDiTActionDecoder`, including candidate value supervision,
  dwell, routing and native noisy-action lift.
- Restored the 200+1000-step execution/transition opening schedule, value loss and
  unboosted main action objective.
- Replaced custom observation with extracted Flow-DINO/soft-address logic and
  restored 49 fine candidates/high-frequency detail.
- Kept inactive prototypes outside the import/source/optimizer closure.

【观察到的结果】

Production configuration became about `182.27M total / 168.09M trainable`, close
to V120 trainable capacity. Flow, 49-point addresses, raw detail, 512 rows and
Evidence MMDiT all executed in focused tests.

【当时的解释】

The earlier 54M total-parameter gap was mostly frozen ancestry; active V120
Flow-DINO+P1 had about 9.7M trainable parameters while the weaker replacement had
about 21.7M. Dataflow, not raw size, explained capability.

【后来怎么看（截至冻结终点）】

Partly retained. The module extraction was real, but the first adapter did not
preserve their input distribution or optimization behavior.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；核心抽取有效，Schema20 集成未通过行为验收。

【证据与锚点】

- 类型：测试/运行观测；历史检索；推断/未决
- 对话：visible messages 3253-3275 and 3331-3333。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`182.27M total / 168.09M trainable`。

【关联】

Restored V120 core, active versus frozen capacity and Schema20.

### CI-0263 — Restored modules were immediately endangered by duplicate and bypassing ingress

【时间 / 版本】

Schema20 adapter review, 2026-08-11.

【当时在解决什么问题】

The V120 decoder class was active, but current top outputs entered it through a
new wrapper.

【当时的 insight / hypothesis】

Behavioral fidelity is determined by the ingress values as much as by the decoder
class. The wrapper duplicated protected consequence as both trajectory and detail,
resent full S/history into the bottom intent compiler, and initially connected
only the last of two flow intervals to G.

【具体做法】

- Restored the V120 object-mainline rule: generic trajectory exact zero and
  protected detail written once.
- Removed full S/history bypass, retaining only current state and last executed
  action at the corresponding bottom boundary.
- Froze unused trajectory adapters whose bias could recreate evidence from zero.
- Ensured both history motion intervals were aligned and available upstream.

【观察到的结果】

Focused tests established a single protected-consequence write and a real generic
trajectory null. They did not yet prove matching carrier scale under training.

【当时的解释】

An exact legacy block wrapped in a changed source competition is a new model.

【后来怎么看（截至冻结终点）】

Retained as a principle. Schema20's later run showed that additional ingress-scale
and temporal differences still remained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；当时修复不充分。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 3274-3284。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。

【关联】

Top-to-bottom evidence path and single-carrier ownership.

### CI-0264 — The flow-unit audit contained and corrected its own wrong intermediate conclusion

【时间 / 版本】

Schema20 history-flow adapter audit, 2026-08-11.

【当时在解决什么问题】

The adapter needed to convert V120 `patch_flow_forward/backward` into normalized
sampling coordinates.

【当时的 insight / hypothesis】

The first reading claimed exported flow was in native DINO-patch pixels and
therefore current code over-amplified it. Reading the producer later showed V120
had already multiplied by `(8-1)/(high_side-1)`: exported flow was in 8×8 chart
cells, and the attempted correction instead under-scaled it.

【具体做法】

- Re-read the flow producer, not only the adapter consumer.
- Replaced native-side conversion with `2/(chart_side-1)` while preserving
  direction reindexing and sign.
- Kept two-step `-8→-4→0` and direct `-4→0` alignment.

【观察到的结果】

For a 16×16 cache, the wrong second conversion would retain only `7/15` of the
proper amplitude (the earlier 24×24 fixture implied `7/23`).

【当时的解释】

Units belong to the producer/consumer pair. An apparently careful local fix can
reverse the error if the tensor was already normalized upstream.

【后来怎么看（截至冻结终点）】

Retained, including the self-correction chain; the initial native-unit claim is
explicitly invalidated.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：初始结论已否定；修正后的 chart-cell contract 保留。

【证据与锚点】

- 类型：推断/未决
- 对话：visible messages 3285-3287 and 3300。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`patch_flow_forward/backward`、`(8-1)/(high_side-1)`。
- 直接量值/表达式（卡内原文）：`2/(chart_side-1)`。

【关联】

Flow horizon units and producer-to-consumer review.

### CI-0265 — Learned existence was not physical validity and could not gate the only fact entrance

【时间 / 版本】

Schema20 P1-prior audit, 2026-08-11.

【当时在解决什么问题】

V120 learned existence varied widely while camera/physical validity was always
one; the adapted P1 multiplied existence into its object prior.

【当时的 insight / hypothesis】

Learned belief about a global object is a diagnostic or routing preference, not
proof that valid local visual evidence is absent. Multiplying it into the only
P1 fact path repeats the shared-null attenuation failure.

【具体做法】

Removed existence from P1's zero mask; retained only physical validity and actual
local support. Object/null competition remained content-conditioned and existence
remained diagnostic.

【观察到的结果】

The historical existence range was about `0.23-0.63` while physical validity was
`1.0`, demonstrating that the two fields had distinct meanings.

【当时的解释】

Confidence, observability and loss support must not share one “validity” field.

【后来怎么看（截至冻结终点）】

Retained and later generalized into the split between current loss support and
future selector validity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible message 3288。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。

【关联】

Schema23 support/selector split and protected P1 facts.

### CI-0266 — V120 interval-loss mass included direct chronological transition supervision

【时间 / 版本】

Schema20 loss-budget replay, 2026-08-11.

【当时在解决什么问题】

The current mainline assigned the whole external `0.02` interval budget to easy
G/S/recognizer/coarse scaffolds and omitted V120's direct semantic transition
half.

【当时的 insight / hypothesis】

Preserving a total coefficient while reallocating its internal owners changes
which representation must carry time.

【具体做法】

Restored the observed 50/50 split between four-interval semantic transition and
the scaffold family; kept newer address/reliability fields inside the existing
future budget rather than increasing total external weight.

【观察到的结果】

The repair reintroduced direct chronological pressure on the fields consumed by
W1/W2 instead of allowing recognizers alone to absorb the interval objective.

【当时的解释】

Loss ownership, not just coefficient sum, is part of the training ABI.

【后来怎么看（截至冻结终点）】

Partly retained. Later exact teacher/S target replays refined which transition
algebra was appropriate.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录；推断/未决
- 对话：visible message 3289。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。

【关联】

S/W chronology and exact weighted-loss ledger.

### CI-0267 — Mainline capacity tests had been validating an inactive operator

【时间 / 版本】

Schema20 bottom review, 2026-08-11.

【当时在解决什么问题】

Green tests asserted `capacity=0` meant zero block update and `capacity=1` meant
full identity, but the active restored V120 path used a different operator.

【当时的 insight / hypothesis】

In active V120 execution, route/dwell chooses whether and how often a block runs;
capacity selects how much ordered low-rank subspace survives inside that block.
Zero capacity does not mean the whole block is skipped.

【具体做法】

- Removed inactive `NestedCapacityOperator` semantics from mainline acceptance.
- Added active bank/decoder non-expansion, warmup and behavior-equivalence tests.
- Corrected `execution_no_updates` to return the pre-block prefix velocity rather
  than using neutral schedule plus capacity zero.

【观察到的结果】

Two previously reassuring tests were shown to prove the wrong object. The new
ablation acquired its literal causal meaning.

【当时的解释】

Tests can create false safety when their named contract is detached from the
runtime call graph.

【后来怎么看（截至冻结终点）】

Retained as a canonical test-provenance failure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测；因果干预
- 对话：visible messages 3303-3308。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`execution_no_updates`。

【关联】

Active source closure and causal ablation naming.

### CI-0268 — Transition selector geometry had become an uncentered second value path

【时间 / 版本】

Schema20 bottom-ingress audit, 2026-08-11.

【当时在解决什么问题】

The adapter passed `transition.selector` as a rollout token and centered
`transition.value` as transition memory. It was unclear whether rollout used the
selector only as a key.

【当时的 insight / hypothesis】

Source inspection showed rollout produced both K and V, so selector content wrote
a second W/current value that bypassed transition zero semantics.

【具体做法】

Kept the 512-row selector as selection geometry but set rollout V to exact zero;
retained centered transition value and protected consequence as the only value
sources.

【观察到的结果】

A nominal key field stopped acting as a duplicate value carrier.

【当时的解释】

Names like selector/key do not constrain the actual K/V behavior of their
consumer.

【后来怎么看（截至冻结终点）】

Retained as a confirmed duplicate-carrier defect.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在当时修复。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 3309-3311。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`transition.selector`、`transition.value`。

【关联】

Single transition owner and key/value provenance.

### CI-0269 — Source identity and static typing needed an explicit vendored boundary

【时间 / 版本】

Schema20 source-closure review, 2026-08-11.

【当时在解决什么问题】

The active model borrowed dataclasses/tools from inactive prototype files, the
checkpoint hash included every `.py` in mainline, and full-tree Pyright reported
hundreds of errors in mechanically extracted dynamic V120 code.

【当时的 insight / hypothesis】

Runtime identity should follow the recursive import closure. Mature vendored
numeric code can be an explicitly untyped boundary; forcing type cleanup through
it risks behavior changes, while leaving adapters as `object` hides real errors.

【具体做法】

- Extracted pure contract/sampling/query types into small active files.
- Built source snapshots from `train.py`'s import closure.
- Added minimal Protocol/casts at adapters without changing numeric kernels.
- Scoped strict typing to integration boundaries and classified the V120 core as
  vendored dynamic code.

【观察到的结果】

The active closure contained 72 files, 19 from V120 core; inactive prototypes did
not affect source digest or optimizer. Adapter/top/runtime type errors reached
zero, while 314 legacy-core warnings were not “fixed” by rewriting math.

【当时的解释】

Source closure, type boundary and behavior preservation are compatible only when
their scopes are explicit.

【后来怎么看（截至冻结终点）】

Retained as an engineering improvement; it did not prove model recovery.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测；推断/未决
- 对话：visible messages 3292-3299 and 3306-3333。
- 段级工件：见 Segment 027 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`train.py`。

【关联】

Checkpoint identity, vendored V120 core and inactive prototypes.

## Segment 027 synthesis

Schema20 corrected the most obvious Schema19 mistake by creating a verified V120
source core and actually instantiating mature visual, P1, transition and Evidence
bottom modules. The implementation review nevertheless found that behavior could
still drift in the adapters: duplicated values, bottom history bypasses, wrong
flow units, confidence masquerading as validity, loss-owner reallocation, tests
targeting an inactive capacity operator and selector fields writing hidden values.
This segment therefore sharpened “mechanical extraction” into “module plus exact
producer/consumer seam,” while still leaving production optimization behavior
unverified.

## Segment 028 — Schema20 early-run failure and reset to a pure unresolved-issue ledger

### Segment boundary

```text
Date:                 2026-08-11 to 2026-08-12
Visible messages:     3370-3440
Starts with:          early Schema20 runtime and gradient comparison against V120
Ends with:            reset to a short ledger of only unresolved source-backed issues
Primary historical artifacts:
  Schema20 early/full-epoch logs
  V120 aligned metrics and source
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0270 — Schema20 restored modules but not V120 optimization dynamics

【时间 / 版本】

Schema20 batch 340, 2026-08-11.

【当时在解决什么问题】

The run used about 11 GB and approximately 1.32 s/batch; determine whether its
fast, low-memory behavior was healthy and whether recovery was on track.

【当时的 insight / hypothesis】

The modules were genuinely present, but their carrier amplitudes and gradient
competition differed enough that the same nominal LR and global clip produced a
different effective optimizer.

【具体做法】

Aligned batch 340 to V120 on action/native/arm/gripper flow, decoded action,
learning rate, owner gradients and S/W/P values.

【观察到的结果】

- Action flow `0.895` versus V120 `0.516` (+73%).
- Arm flow `0.870` versus `0.431`; gripper was similar (`1.050` versus `1.029`).
- Decoded action was close (`0.114` versus `0.109`).
- Preclip norm was `9.67` versus V120 `1.59`, about 6.1×.
- G pair cosine reached `0.711`, near V120 `0.707`; W was learning.

【当时的解释】

The failure was not missing blocks or data. The adapted top-to-bottom distribution
made a few owners dominate the global clip, slowing the arm/action path despite
similar nominal settings.

【后来怎么看（截至冻结终点）】

Retained as the decisive Schema20 falsification.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3376-3389。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。

【关联】

Gradient spikes, owner competition and behavior-equivalent optimization.

### CI-0271 — A common S temporal base and transition together monopolized the action path

【时间 / 版本】

Schema20 source/log diagnosis, 2026-08-11.

【当时在解决什么问题】

Explain the 6× larger global gradient and slow arm learning despite active G/W.

【当时的 insight / hypothesis】

S temporal variation was nearly zero, yet P3 copied its high-RMS common temporal
source directly as an unconditional additive base. The modified transition then
became the other dominant action carrier.

【具体做法】

Compared common versus consequence-conditioned temporal RMS and owner gradient
shares after the global clip.

【观察到的结果】

- S interval variation stayed around `0.0163`; true temporal variation was about
  `8.4e-6` versus V120 `0.021`.
- P3 temporal base was `0.516`; temporal×consequence interaction only `0.00547`,
  so W affected about 1% of that lane.
- At batch 340 transition gradient was `0.891` after clipping, about 79% of squared
  global gradient mass; at batch 20 bottom heads occupied about 96%.

【当时的解释】

Global clipping was not the primary bug; it exposed a scale/ownership bug. One
common S path and transition captured the effective learning-rate budget while
structured owners were uniformly shrunk.

【后来怎么看（截至冻结终点）】

Retained. Later work repeatedly revisited S publicization and transition/P2
ownership.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；具体 solutions later changed.

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 3382-3395。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。

【关联】

S public/typed split, transition gradient spikes and global clipping.

### CI-0272 — A static cached transition was not V120's per-ODE action-conditioned dynamics

【时间 / 版本】

Schema20 source replay, 2026-08-12.

【当时在解决什么问题】

The current transition retained 512 rows and a centered value, so it had been
described as restored.

【当时的 insight / hypothesis】

Its lifecycle was different: current code built it in `encode_online()` from a
history proposal and cached it for all five steps, whereas V120 transition read
current trajectory/action tokens at every ODE step.

【具体做法】

Traced construction frequency, proposal source and downstream consumer rather
than checking only row count and zero semantics.

【观察到的结果】

The current path was a static history-conditioned evidence bank, not dynamic
action-conditioned transition. This explained some speedup and its need to
compensate through large action gradients.

【当时的解释】

Lifecycle and conditioning state are part of tensor meaning. A cached tensor with
the same shape cannot substitute for a per-step field.

【后来怎么看（截至冻结终点）】

Retained as a confirmed regression and later deployment-lifecycle test target.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible message 3415。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`encode_online()`。

【关联】

Static/dynamic P1-P3 split and five-step endpoint lifecycle.

### CI-0273 — The “earlier detail” target paired one feature frame with another frame's RGB

【时间 / 版本】

Schema20 observation-target audit, 2026-08-12.

【当时在解决什么问题】

The adapter exposed earlier and previous high-resolution detail for two temporal
geometry losses.

【当时的 insight / hypothesis】

`earlier_detail_features` was simply assigned from `previous_detail_features`,
while `earlier_literal_rgb` came from a genuinely earlier frame. The target
therefore compared the same feature activation under two different RGB labels.

【具体做法】

Classified the defect as certain but deferred the implementation choice because
holding a third high-resolution activation had memory consequences. The low-risk
proposal was to remove the false feature loss and retain genuine DINO/flow/RGB
supervision for the older interval.

【观察到的结果】

This was a source-proven label/feature mismatch independent of long-run behavior.

【当时的解释】

Duplicating a tensor to satisfy an interface invents temporal evidence; it is not
a harmless memory optimization.

【后来怎么看（截至冻结终点）】

Retained as a confirmed historical defect. The frozen archive contains no
independent evidence that this exact target mismatch was later repaired, so its
historical closure state remains unknown rather than an active current claim.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留；确定性历史故障，冻结档案未记录独立修复闭环。

【证据与锚点】

- 类型：历史检索；推断/未决
- 对话：visible message 3417 and 3423。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`earlier_detail_features`、`previous_detail_features`、`earlier_literal_rgb`。

【关联】

Temporal provenance and high-resolution activation memory.

### CI-0274 — Architecture contracts can preserve a mistake if implementation is confused with validation

【时间 / 版本】

Post-Schema20 review, 2026-08-11/12.

【当时在解决什么问题】

The compact architecture contract declared public P3 temporal base and
real-minus-zero transition as non-negotiable invariants even though early logs
identified them as the most suspicious paths.

【当时的 insight / hypothesis】

A contract should describe an accepted behavior decision, not promote every
implemented experiment into truth. Otherwise later review treats the defect as
an invariant and designs compensating mechanisms around it.

【具体做法】

- Separated “implemented” from “validated.”
- Reclassified current items as confirmed harmful, source-restored but
  behavior-unverified, or proven engineering improvements.
- Added an explicit recovery warning rather than modifying the model during the
  ledger pass.

【观察到的结果】

The current contract itself was recognized as part of the audit failure, not an
independent proof of correctness.

【当时的解释】

Documentation can create confirmation bias when it is written before empirical
and source-level closure.

【后来怎么看（截至冻结终点）】

Retained and later reflected in the AGENTS rule that the compact contract is
authority only after architectural adoption, while ancestry stays auxiliary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；历史检索
- 对话：visible messages 3402-3418。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。

【关联】

Architecture memory governance and historical versus active authority.

### CI-0275 — The current issue ledger was redesigned to contain only unresolved problems

【时间 / 版本】

Issue-governance reset, 2026-08-12.

【当时在解决什么问题】

The old ledger mixed historical plans, completion claims and still-active
problems, allowing a code change or passing test to erase a behavior failure.

【当时的 insight / hypothesis】

A short operational ledger should list only live problems and delete each item
after source, behavior and acceptance closure. Detailed history belongs elsewhere.
Priority labels do not imply causal certainty; evidence level must be explicit.

【具体做法】

- Created `CURRENT_MAINLINE_ISSUES.md` with source evidence, log evidence, impact,
  repair boundary and acceptance condition.
- Started with five P0/five P1, then performed a second reverse audit instead of
  claiming completeness and added two missed P1 issues.
- Classified items as confirmed fault, confirmed degradation, unverified V120
  deviation or upstream-dependent risk.
- Locked the baseline to exact V120 commit/log/script identity and excluded
  `v120_begin.log` and `v120_long_pre_fix.log`.

【观察到的结果】

The second pass found two real omissions: S trained against four co-moving,
potentially low-distinction factorized targets, and P1 queries received overlapping
history causality through S temporal, coarse action and a 24-step history proposal
while address variation kept falling.

【当时的解释】

An issue list is not complete merely because it is structured. Independent
producer→consumer and consumer→producer passes are necessary, and upstream P0
failures can still hide later secondary defects.

【后来怎么看（截至冻结终点）】

Retained. The pure-problem ledger became a recurring user requirement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 3418-3436。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`CURRENT_MAINLINE_ISSUES.md`、`v120_begin.log`、`v120_long_pre_fix.log`。

【关联】

Mandatory subsystem familiarity, reverse review and evidence grading.

### CI-0276 — User frustration changed the technical acceptance process, not the evidence itself

【时间 / 版本】

Schema20 failure discussion, 2026-08-11/12.

【当时在解决什么问题】

Repeated claims of local closure were followed by another early run far behind
V120. The user challenged why optimization/refactoring had changed inputs,
outputs and gradient geometry without prior justification.

【当时的 insight / hypothesis】

The technical response became stricter: any structural change had to state its
source/log evidence, invariants, expected RMS/Jacobian/gradient-share change,
shortcut risks and rejection criterion before implementation. Static dataflow
review should catch obvious public bypasses instead of adding adapters or losses.

【具体做法】

- Stopped model modification and wrote the issue ledger first.
- Explicitly withdrew claims that local V120-level metrics or engineering gains
  constituted model improvement.
- Materialized the full V120 snapshot for ordinary source comparison.

【观察到的结果】

The assistant acknowledged that no model capability had yet been reliably shown
to exceed V120; lower W cosine was not correctness, and speed/memory were
engineering rather than policy gains.

【当时的解释】

The interaction forced a necessary distinction between “code present,” “metric
looks different,” “baseline recovered” and “new model benefit.” The user's tone
is not technical evidence; the resulting review requirements are.

【后来怎么看（截至冻结终点）】

Retained as a process-changing event and recorded without using anger as a causal
claim about the model.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 3386-3409 and 3412-3439。
- 段级工件：见 Segment 028 boundary 的 `Primary historical artifacts`。

【关联】

Subsystem-familiarity rule, issue-ledger discipline and trust repair.

## Segment 028 synthesis

Schema20 demonstrated a second kind of failed recovery: real V120 modules can be
present while wrappers, lifecycle and scale redefine their behavior. A common S
temporal base and an altered transition dominated the global gradient budget,
making nominally identical optimization much slower. Source replay then uncovered
a static/dynamic transition mismatch and a false earlier-detail target. The
conversation responded by demoting unverified contract text, creating a pure
unresolved-problem ledger, grading evidence and requiring both forward and reverse
dataflow review before changes. The exact V120 final run—not its pre-fix logs—was
locked as the default behavior baseline.

## Segment 029 — Full-source correction, Schema21 and the dynamic-P1 boundary failure

### Segment boundary

```text
Date:                 2026-08-12
Visible messages:     3441-3516
Starts with:          direct current-versus-V120 full-source replay and self-correction
Ends with:            Schema21 evidence that the dynamic P1 factual boundary remained absent
Primary historical artifacts:
  complete V120 source snapshot
  Schema20 completed log
  Schema21/22 P1-P3 adapters, losses and validation lifecycle
```

### CI-0277 — V120 recovery had to preserve execution order as well as modules and axes

【时间 / 版本】

Schema21 source reconstruction, 2026-08-12.

【当时在解决什么问题】

Schema20 had restored 49 P1 candidates, five P lanes, 512 transition rows and the
V120 bottom, but not their exact call order or query state.

【当时的 insight / hypothesis】

The V120 path was sequential: P1 read clean action-basis identity and wrote the
trajectory; P2 read `seed+P1`; P3 read `seed+P1+P2`; transition then consumed the
terminal-normalized full trajectory. Moving transition earlier or using a history
proposal changed both its input semantics and gradients.

【具体做法】

- Restored P1 query-specific selection over 49 candidates.
- Restored W1/W2 four-interval consequences and five P3 lanes.
- Restored default EvidenceViewAdapter, separate current/history state, native
  physical action tokens and `transition.value + event_context`.
- Restored dynamic per-ODE transition after P writes.

【观察到的结果】

The mainline could execute the reconstructed sequence under BF16 and tests, but
later source comparison still found missing P1/P2 adapter heads.

【当时的解释】

The consumer-visible state at a call site is part of an operator; executing the
same class earlier in the graph is not equivalent.

【后来怎么看（截至冻结终点）】

Retained as a durable lifecycle/ordering rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3441-3448 and 3460-3467。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。

【关联】

Five-step dynamic lifecycle and producer/consumer sequencing.

### CI-0278 — Direct source replay repeatedly corrected plausible but false audit conclusions

【时间 / 版本】

Schema21 three-pass source audit, 2026-08-12.

【当时在解决什么问题】

Several local readings appeared to identify V120 behavior but contradicted deeper
call-chain evidence.

【当时的 insight / hypothesis】

Three notable claims were revised:

1. Transition was first said to consume 24 horizon tokens; the real V120 consumer
   used the post-P, normalized `24×4=96` trajectory rows.
2. History proposal appeared to enter generic canvas memory; the formal
   `object_intent_dynamics_323` path masked it in G, used explicit W/P interfaces
   and zeroed proposal trajectory at bottom, leaving only auxiliary supervision.
3. Earlier detail was suspected to alias previous detail; the exact V120 path
   intentionally retained only the last two high-resolution frames and represented
   the earlier interval through the native two-pair flow ledger.

【具体做法】

Followed each apparent producer through capability masks, active adapters and
terminal consumers before editing; corrected comments/tests rather than adding
the proposed restorations.

【观察到的结果】

The full snapshot prevented three incorrect “fixes” that would have changed token
normalization, reintroduced a masked proposal bypass or retained a false third
high-resolution activation.

【当时的解释】

Function signatures and generic trunks overstate what a capability actually
consumes. Active masks and downstream neutralization can invalidate a local
appearance of use.

【后来怎么看（截至冻结终点）】

Retained as an important self-correction chain.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：三项初步结论已修正；审查方法保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 3457-3462 and 3471-3473。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`object_intent_dynamics_323`。

【关联】

Complete source snapshot, capability masks and historical target provenance.

### CI-0279 — P1/P2 layer-contract adapters and the shared seed were active policy machinery

【时间 / 版本】

Schema21 source replay, 2026-08-12.

【当时在解决什么问题】

The refactored mainline replaced two V120 `LayerContractAdapterHeads` with
dictionary assembly and separately projected state/history for transition and
bottom.

【当时的 insight / hypothesis】

The adapters were not diagnostic heads. P1 and P2 contracts wrote different
trajectory states into bottom-readable state/history/rollout, shared one
action/context seed, and P2 event evidence entered Evidence MMDiT directly.

【具体做法】

- Restored one shared V120 seed.
- Restored distinct P1-after and P2-after layer contracts.
- Restored P2 event output to EvidenceView.
- Logged each contract and seed gradient separately.

【观察到的结果】

The two restored contracts accounted for exact known parameter inventories
(`1,876,270` and `265,472`), and focused tests proved they altered the real
Evidence MMDiT ingress.

【当时的解释】

An adapter can contain essential learned distribution alignment even if its
output fields resemble a simple typed dictionary.

【后来怎么看（截至冻结终点）】

Retained as a confirmed omission from the independent rewrite.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在 Schema21 恢复。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3463-3470。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。

【关联】

Shared canvas seed, bottom ingress and layer-contract behavior.

### CI-0280 — A V120 directional loss had a nonzero floor even at perfect prediction

【时间 / 版本】

Schema21 loss audit, 2026-08-12.

【当时在解决什么问题】

The V120 “direction loss” used floor-regularized normalization whose self-dot was
less than one, so `prediction == target` still produced positive loss.

【当时的 insight / hypothesis】

This was a static, source-proven V120 defect that could be fixed without inventing
a new target or changing external weighting.

【具体做法】

Replaced it with squared difference in the same smoothly normalized space while
preserving the scale floor and external coefficient. Restored the actual V120
internal loss mix `0.30/0.25/0.15/0.05/0.08/0.07/0.10` and removed unverified
extra reliability/address pressure.

【观察到的结果】

The objective could reach zero at equality and kept the intended smoothness.

【当时的解释】

The user's rule permitted correcting demonstrable baseline defects; “default to
V120” did not mean preserving a mathematically impossible optimum.

【后来怎么看（截至冻结终点）】

Partly retained. Later exact-Teacher recovery changed some surrounding target
semantics, but the zero-at-equality principle remained valid.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：确认的 V120 缺陷；当时修复。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 3447-3450。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`0.30/0.25/0.15/0.05/0.08/0.07/0.10`。

【关联】

Evidence-graded deviations from V120.

### CI-0281 — Loss-required execution tensors were incorrectly tied to a diagnostic batch budget

【时间 / 版本】

Schema20 validation crash found during Schema21 review, 2026-08-12.

【当时在解决什么问题】

Schema20 completed 2,840 training batches but validation crashed on its fifth
batch with missing execution-supervision tensors.

【当时的 insight / hypothesis】

`eval_diagnostic_batches=4` caused the first four validation batches to construct
candidate-value tensors. Batch five disabled diagnostics, but execution-value
loss still required those tensors. Training/loss state had been incorrectly
classified as optional telemetry.

【具体做法】

- Split mandatory train/eval supervision tensors from optional detailed logging.
- Kept ordinary five-step deployment free of loss-only candidate tensors.
- Added a full `eval_step()` regression with diagnostics disabled.

【观察到的结果】

The exact fifth-batch failure was reproduced and fixed. This was not OOM, NaN,
BF16 instability or training divergence.

【当时的解释】

Diagnostic frequency may control observation cost, never tensors required to
define the validation objective.

【后来怎么看（截至冻结终点）】

Retained as a confirmed runtime-boundary error.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3480-3489。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`eval_step()`。

【关联】

Loss versus diagnostics lifecycle and validation completeness.

### CI-0282 — The completed Schema20 epoch exposed a timed null-absorption cascade

【时间 / 版本】

Schema20 epoch-one training, 2026-08-12.

【当时在解决什么问题】

The first partial audit ended at batch 340; the complete pre-validation log could
show whether early warnings persisted or changed phase.

【当时的 insight / hypothesis】

P2 inherited P1 object posterior and null prior, creating positive feedback:
increasing P1 null lowered P2 candidate prior, lowered effect/action gradients,
and allowed null to rise further.

【具体做法】

Reconstructed the trajectory of semantic/geometry null, P2 effect, consequence
gradient, transition gradient, P1 variation, P3 lanes, gripper events, flow and
execution ranking over all 2,840 batches.

【观察到的结果】

- P2 null was `0.038/0.076` at batch 20, fell early, then jumped near `0.997` by
  batch 420 and about `0.999995` by batch 2,840.
- P2 effect peaked near `0.282` then reached `2.22e-7`.
- Consequence gradient fell from `6.99e-4` at batch 200 to `8.99e-13` at the end.
- Transition tail postclip median was `0.853`, with preclip peaks to `25.89`.
- G/W object identity separated, but W adjacent interval cosine remained `0.982`.
- Gripper event error remained about 3.26× hold error.

【当时的解释】

The run was trainable and learned object identity, but a specific posterior
feedback loop removed W→P action pressure and left temporal/transition shortcuts.

【后来怎么看（截至冻结终点）】

Retained. It refined rather than replaced the batch-340 Schema20 diagnosis.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible message 3489。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。

【关联】

Null absorption, shared priors and action-level consequence gradients.

### CI-0283 — Schema21's remaining failure was the replacement of dynamic P1 with a weak static aggregate

【时间 / 版本】

Schema21 batches 20-600, 2026-08-12.

【当时在解决什么问题】

Schema21 removed transition domination and retained healthy visual/G/object axes,
yet arm flow fell behind V120 after batch 100.

【当时的 insight / hypothesis】

V120 factual P1 combined protected policy detail with the per-ODE P1 policy-block
innovation. Schema21 used a static reader's `aggregate_fact` as the complete
factual base, allowing signed object content to cancel raw detail and learned null
to attenuate it before P2/P3.

【具体做法】

Aligned Schema21 and V120 at batches 20-600 and traced P1 detail, completed fact,
P2 effect, consequence/P3 gradients and runtime call frequency.

【观察到的结果】

- Batch 260-340 action flow was 31.2% higher and arm flow 42.6% higher than V120,
  while native flow differed only 1.6% and gripper/decoded action were not worse.
- At batch 340 factual base was about `0.044-0.047`, at least 7× below V120.
- By batch 600, detail remained `0.0525` but factual base was about `0.0023`;
  V120 factual/precision were about `0.296/0.252`.
- Consequence gradient reached about `2.1e-6`; temporal remained the available lane.

【当时的解释】

The model read visual detail but did not perform the dynamic policy write that
made it action-specific and correctly scaled. Faster runtime partly reflected
missing per-step computation, not only efficient caching.

【后来怎么看（截至冻结终点）】

Retained as the direct motivation for Schema22, though Schema22 itself was later
experimentally rejected for additional behavioral mismatches.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；Schema22 attempted repair.

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3492-3510。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`aggregate_fact`。

【关联】

Static/dynamic P1 ownership and policy-specific factual write.

### CI-0284 — Schema22 restored dynamic P1 without reopening high-resolution vision at each ODE step

【时间 / 版本】

Schema22 freeze, 2026-08-12.

【当时在解决什么问题】

Repair the dynamic P1 factual boundary without multiplying the expensive 49-point
high-resolution read five times.

【当时的 insight / hypothesis】

Static factual evidence and dynamic policy writing are different lifecycle
owners. High-resolution detail can be read once per observation; a compact P1
policy block can write action/time-conditioned residuals at each Euler node.

【具体做法】

- Kept one static high-resolution/49-candidate read.
- Restored dynamic P1 execution at every ODE step.
- Prevented signed public object value and learned null from deleting protected
  detail.
- Routed P2/P3 through the completed dynamic P1 fact.
- Added owner/logging fields and rejected exact Schema21 resume.

【观察到的结果】

136 targeted tests and CPU BF16 end-to-end backward passed with finite nonzero
dynamic-P1 gradients. No Schema22 production evidence existed yet.

【当时的解释】

The repair separated an expensive static producer from a cheap dynamic consumer
while preserving the policy-dependent write.

【后来怎么看（截至冻结终点）】

Partly retained. Schema22 was later rejected as a behavior baseline, but the
static-fact/dynamic-policy ownership distinction remained central.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；Schema22 后续被实验拒绝。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 3508-3516。
- 段级工件：见 Segment 029 boundary 的 `Primary historical artifacts`。

【关联】

CompletedP1PolicyState and later factual-base/policy-residual interfaces.

## Segment 029 synthesis

Direct file-to-file replay corrected several plausible audit errors and restored
active V120 sequencing, layer-contract heads and shared seed behavior. It also
found a runtime crash caused by conflating loss state with optional diagnostics.
The complete Schema20 log exposed a null-absorption phase transition, while early
Schema21 evidence isolated a different, simpler regression: the expensive static
detail read survived, but the dynamic P1 policy write and protected factual scale
did not. Schema22 responded by splitting static high-resolution evidence from a
per-ODE dynamic P1 residual. Local closure was achieved, but later experiments
still had to determine whether the wider V120 behavior contract was faithful.

## Segment 030 — Schema22 rejection and the Schema23 behavior-contract repair

### Segment boundary

```text
Date:                 2026-08-13
Visible messages:     3517-3562
Starts with:          Schema22 early log and the supplied V120 source-audit package
Ends with:            implemented and reviewed Schema23 behavior-contract repair
Primary historical artifacts:
  schema22_long.log
  schema22_v120_source_audit_v3.zip
  schema22_v120_source_audit_v4_corrected.zip
  .audit/v120_exact_source_0b92d359/
  v120_long.log
```

### CI-0285 — Schema22 was numerically rejected, not merely slower than V120

【时间 / 版本】

Schema22 epoch 1-3, 2026-08-13.

【当时在解决什么问题】

Determine whether the dynamic-P1 repair had restored V120 behavior and explain
the new run's termination.

【当时的 insight / hypothesis】

Low local action loss was insufficient evidence of recovery. The complete run
had to be checked for rollout quality, future targets, object identity and the
actual backward failure boundary.

【具体做法】

Parsed the complete Schema22 and V120 logs, aligned 390 common points and
separated the last finite logged forward from the next unlogged backward.

【观察到的结果】

- Schema22 completed two epochs, then the backward after `step=7812` produced a
  non-finite gradient and failed fast.
- The last recorded batch was finite, so the log could not identify the first
  offending parameter.
- Over aligned points, physical flow averaged about `0.1628` versus V120
  `0.1101`; first/tail RMSE and gripper event ratios were substantially worse.
- G and W remained more homogeneous, Teacher semantic/transport targets were
  much smaller and P2 largely abandoned geometry.

【当时的解释】

Schema22 had repaired one local ownership boundary but not the wider training,
Teacher and deployment contract. The non-finite failure made exact continuation
invalid independently of performance.

【后来怎么看（截至冻结终点）】

Retained. Schema22 remained an experimentally rejected ancestor, while several
of its local static/dynamic-P1 ideas survived in later interfaces.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；Schema22 rejected.

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3518-3522 and 3529。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。

【关联】

Schema23 lifecycle repair and non-finite sentinel.

### CI-0286 — Audit provenance must be corrected before its recommendations are used

【时间 / 版本】

Schema22 V3/V4 audit review, 2026-08-13.

【当时在解决什么问题】

The supplied V3 package contained useful aligned series and lifecycle matrices,
but also claimed that the exact V120 implementation was unavailable.

【当时的 insight / hypothesis】

An audit can contain excellent measurements and still have invalid causal
conclusions if its source-provenance premise is wrong. Recommendations must be
regraded once exact source exists.

【具体做法】

- Matched the seven serialized object-module hashes against commit `0b92d359`.
- Later materialized the complete source at
  `.audit/v120_exact_source_0b92d359/`.
- Kept the original V3 package and emitted a separately traceable corrected V4.

【观察到的结果】

The exact object source was available, and the later full snapshot matched all
433 tracked files with zero missing or differing files. Several conclusions
previously labelled C-grade could therefore be checked directly.

【当时的解释】

The error was not lack of evidence but failure to use existing evidence as the
primary local reference.

【后来怎么看（截至冻结终点）】

Retained as a durable provenance rule. Later reviews repeatedly returned to
local snapshots and run contexts instead of reconstructing behavior from names
or isolated Git fragments.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 3523-3524, 3527-3529 and 3582-3587。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`0b92d359`、`.audit/v120_exact_source_0b92d359/`。

【关联】

Experiment identity and source-backed replay.

### CI-0287 — A numerically suspicious dead branch cannot explain a backward failure

【时间 / 版本】

Schema22 audit correction, 2026-08-13.

【当时在解决什么问题】

V3 attributed the non-finite backward to a `grid_sample` and inverse sampled-mass
calculation in `future_address`.

【当时的 insight / hypothesis】

Before diagnosing a dangerous operator, trace its differentiable consumers. A
bad-looking tensor with no loss or routing consumer cannot be the first backward
offender.

【具体做法】

Followed `FutureObjectDynamics.future_address` through both Schema22 and V120,
including construction, concatenation, shape checks, detached diagnostics, P2
and losses.

【观察到的结果】

`future_address` had no differentiable consumer. It was wasted computation and
technical debt, but not a causal path to the failure.

【当时的解释】

Adding a numerical guard to a dead branch would have created the appearance of a
fix without changing the failed backward graph.

【后来怎么看（截至冻结终点）】

Retained as a source-tracing lesson. The field was explicitly left as diagnostic
debt during the behavior repair.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已排除为崩溃根因。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 3525-3526 and 3542。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`grid_sample`、`future_address`、`FutureObjectDynamics.future_address`。

【关联】

Producer-consumer closure and dead-branch audits.

### CI-0288 — Learned null is legitimate only outside the protected factual base

【时间 / 版本】

Schema22 audit correction, 2026-08-13.

【当时在解决什么问题】

The audit alternated between recommending a P1 conditional null and banning
learned null because an earlier design had used it as an escape route.

【当时的 insight / hypothesis】

The correct distinction is not learned versus fixed null. It is whether null can
erase mandatory information or only reject optional innovation.

【具体做法】

The corrected boundary was written as:

```text
P1 = protected_current_base
   + (1 - p_null) * optional_detail_innovation
```

The null key/logit may be learned, but its value must be exact zero; `p_null=1`
must leave the protected factual base bit-exact. Its prior may use current
observable evidence, not noisy action, future Teacher state or a policy carrier.

【观察到的结果】

The overly absolute audit language was corrected. No new learned-null mechanism
was mixed into Schema23.

【当时的解释】

Null should mean “no reliable optional object read,” never “skip P1 because that
is easier for the policy.”

【后来怎么看（截至冻结终点）】

Retained as the project's anti-shortcut null boundary and reused in later
per-lane routing discussions.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；当时未实施新 P1 null。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 3530-3532。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。

【关联】

Protected bases, optional innovation and dustbin/null semantics.

### CI-0289 — Schema23 repaired four behavior contracts without redesigning G/S/W/P

【时间 / 版本】

Schema23 implementation, 2026-08-13.

【当时在解决什么问题】

Restore the V120 lifecycle and Teacher semantics that Schema22 had changed while
keeping the current flow-time coordinate and owned RNG.

【当时的 insight / hypothesis】

The smallest defensible recovery was behavioral: mirrored sampling, exact Euler
and endpoint-head calls, exact Teacher moments, split validity ownership and a
first-offender gradient sentinel.

【具体做法】

- Sampled `t_v120 ~ Beta(1.5,1.0)*.999+.001` and used `t_main=1-t_v120`.
- Restored action updates at `0,.2,.4,.6,.8` plus a head-only forward at `1.0`.
- Restored the 5:1 positive-event weight.
- Removed entropy/reliability shrinkage from successor, transport and covariance.
- Split `current_loss_support` from `future_selector_validity`.
- Checked total gradient norm after backward and before clipping, with structured
  first-parameter failure logging.

【观察到的结果】

The implementation and targeted tests closed these contracts without changing
parameter topology or adding a loss, gate, block or P1 null.

【当时的解释】

These were demonstrated semantic regressions, not capacity or optimization
hypotheses.

【后来怎么看（截至冻结终点）】

Retained. These repairs survived subsequent schemas even though Schema23 as a
whole did not recover V120.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3536-3562。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`current_loss_support`、`future_selector_validity`。

【关联】

V120 lifecycle, Teacher algebra and support/selector ownership.

### CI-0290 — Reverse review found moment, consumer and overflow errors after the first repair

【时间 / 版本】

Schema23 implementation review, 2026-08-13.

【当时在解决什么问题】

The first implementation passed obvious interface checks but still differed
from the intended Teacher and loss-owner semantics.

【当时的 insight / hypothesis】

An interface split is not closed until every consumer is audited, and a global
non-finite norm does not imply that an individual element is non-finite.

【具体做法】

- Removed a per-camera division that converted V120's raw posterior moment into
  a conditional coordinate.
- Prevented the recognizer from consuming either selector validity or the
  future-loss-only support field.
- Extended the sentinel to catch a tensor whose finite FP32 elements produce a
  non-finite tensor norm.

【观察到的结果】

`current_loss_support` was left with only future-dynamics loss consumers, target
selector/reliability could not alter auxiliary recognition loss, and overflow
could still be assigned to the first stable parameter owner.

【当时的解释】

These issues were found only by tracing backward from consumers and failure
reporting, not by checking shapes at the producers.

【后来怎么看（截至冻结终点）】

Retained as an example of why two-direction review is mandatory after a typed
interface change.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 3551, 3554 and 3557-3559。
- 段级工件：见 Segment 030 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`current_loss_support`。

【关联】

Mandatory subsystem-familiarity and reverse dataflow review.

## Segment 030 synthesis

Schema22 showed that a locally correct dynamic-P1 repair could coexist with a
failed wider behavior contract and an unlocalized non-finite backward. Review of
the supplied audit then established three durable disciplines: correct source
provenance before accepting causal claims, trace suspicious operators to real
consumers, and distinguish a legitimate zero-valued optional null from a
shortcut that can erase protected facts. Schema23 restored four demonstrated
V120 contracts and its reverse review caught additional moment, recognizer and
overflow-boundary mistakes. These repairs were real, but they did not yet prove
that the independently rewritten top path was faithful.

## Segment 031 — Schema23 early evidence and complete V120 source-level replay

### Segment boundary

```text
Date:                 2026-08-13
Visible messages:     3563-3638
Starts with:          Schema23 early training evidence
Ends with:            frozen Schema24 faithful-main-path recovery plan
Primary historical artifacts:
  mainline_v120_contract_repair_b8.log
  schema22_v120_source_audit_v4_corrected.zip
  .audit/v120_exact_source_0b92d359/
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0291 — Restored Teacher scale did not restore object identity

【时间 / 版本】

Schema23 batches 20-280, 2026-08-13.

【当时在解决什么问题】

Check whether the four behavior repairs improved the early run and quantify the
remaining weakness against V120 rather than the rejected Schema22.

【当时的 insight / hypothesis】

“W is weak” was too coarse. Magnitude, interval variation, object separation and
actual P2 consumption had to be separated.

【具体做法】

Aligned Schema23 and V120 at batch 280 and compared G correction, object cosine,
S/W interval variation, Teacher target scale, P2 scores/null and P3 output.

【观察到的结果】

- Teacher semantic and transport returned to the V120 scale.
- W interval variation was about 86% of V120 and consequence effect was not
  small.
- G3 parent correction was only about 11.8% of V120.
- W object-separation measure was about 25× weaker, P2 content about 35% of V120
  and null mass about 12× larger.

【当时的解释】

The run did not lack future magnitude. It lacked identifiable object evidence,
so a learned null became the easiest competitor.

【后来怎么看（截至冻结终点）】

Retained as an important diagnostic decomposition. Later source audit found the
missing progressive G and altered P1/grounder pathways upstream of this symptom.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3563-3572。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。

【关联】

G identity, W object specificity and P2 null absorption.

### CI-0292 — Same-iteration trajectories exposed a phase change hidden by endpoint comparison

【时间 / 版本】

Schema23 epoch-one batches 20-2200, 2026-08-13.

【当时在解决什么问题】

Explain why an apparently healthy early run later fell behind V120.

【当时的 insight / hypothesis】

The direct action path could optimize quickly while the future-effect path lost
its contribution. A single final point could not reveal the transition.

【具体做法】

Aligned every available iteration and split the run into early, middle and late
windows, tracking action flow, P2 null and effect magnitude together.

【观察到的结果】

Schema23 was sometimes 24-51% better in action flow around batches 500-1000, but
V120 crossed it around 1200-1500. In batches 1520-2200, action flow was about
22.4% worse while P2 null was `0.718` versus `0.498` and effect magnitude was
`0.0347` versus `0.0500`.

【当时的解释】

Early action optimization was real, but it did not acquire continued structured
future-effect bandwidth. Null absorption removed the later marginal gain.

【后来怎么看（截至冻结终点）】

Retained. Similar “early fit, later structural ceiling” patterns recurred and
motivated windowed rather than endpoint-only audits.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3573-3578。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。

【关联】

Training phase transitions and aligned-window comparisons.

### CI-0293 — A local complete snapshot became the sole ancestry reference

【时间 / 版本】

Schema23 full-source audit, 2026-08-13.

【当时在解决什么问题】

The review repeatedly returned to Git fragments despite the user's supplied
audit package and local V120 material.

【当时的 insight / hypothesis】

Once a verified complete local source snapshot exists, ancestry review should
use it consistently. Switching evidence sources wastes time and risks mixing
partial revisions.

【具体做法】

Stopped fragmentary Git inspection, unpacked the corrected audit and completed
the missing runtime/trunk coverage in `.audit/v120_exact_source_0b92d359/`.

【观察到的结果】

The complete tree immediately exposed omissions that the package-level audit
had missed, including decoder-local clipping and the active G/P1 call structure.

【当时的解释】

The technical failure was source selection, not source availability.

【后来怎么看（截至冻结终点）】

Retained. The user's strong correction changed the review method: local complete
artifacts and manifests had to be exhausted before historical reconstruction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 3579-3593。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`.audit/v120_exact_source_0b92d359/`。

【关联】

CI-0286 and evidence-source discipline.

### CI-0294 — Fake camera geometry created a deterministic preference for null

【时间 / 版本】

Schema23 full-source audit, 2026-08-13.

【当时在解决什么问题】

Identify why P2 increasingly rejected real W effects even though the action edge
and future magnitude existed.

【当时的 insight / hypothesis】

An axis in the type signature is not real if it was first collapsed and later
restored by `expand`. Downstream averaging of squared distances then adds a
systematic penalty unavailable to null.

【具体做法】

Traced Teacher/W/P2 geometry. W predicted `[B,I,K,2/3]`, expanded it to camera,
while P2 averaged camera-wise squared distance.

【观察到的结果】

The real-route score decomposed into distance from the mean camera coordinate
plus an always-nonnegative camera-coordinate variance penalty. Null retained a
fixed zero logit and therefore became increasingly favorable.

【当时的解释】

This was not a missing gradient edge. The connected downstream route was
rationally rejecting an invalid geometric distribution supplied upstream.

【后来怎么看（截至冻结终点）】

Retained as a confirmed structural defect. Schema24 removed fake camera by
returning to object-level geometry, before a later design again revisited real
camera-preserving geometry.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；Schema24 addressed it.

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3589-3593 and 3736-3742。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`[B,I,K,2/3]`。

【关联】

P2 null routing and axis conservation.

### CI-0295 — Optimizer and gradient lifecycle are part of the architecture behavior contract

【时间 / 版本】

Schema23 full-source audit, 2026-08-13.

【当时在解决什么问题】

Explain differences that could not be assigned to G/W/P math and determine
whether gradient metrics were comparable to V120.

【当时的 insight / hypothesis】

Weight-decay membership, local clipping order and metric sampling phase change
optimization trajectories even when forward modules are identical.

【具体做法】

- Compared V120's decoder-local clip before global clip with the current global-
  only lifecycle.
- Audited AdamW parameter-name ownership rather than assuming all bias/1D tensors
  should be no-decay.
- Compared backward-time raw diagnostics with post-global diagnostics.

【观察到的结果】

Schema23 lacked V120's decoder-local clip, moved many ordinary bias/LayerNorm
parameters to no-decay and logged gradients after global clipping. Existing
gradient magnitudes therefore could not be compared directly.

【当时的解释】

These were behavioral differences, not logging cosmetics. The audit tool also
needed raw/post-local/post-global semantics to avoid calling clipped values raw.

【后来怎么看（截至冻结终点）】

Retained. Schema24 restored the lifecycle and the three-stage audit names.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3587, 3605-3612 and 3635。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。

【关联】

Optimizer ownership, gradient spikes and recovery-audit comparability.

### CI-0296 — The independent mainline had replaced two active V120 computations, not reorganized them

【时间 / 版本】

Schema23 full-source audit, 2026-08-13.

【当时在解决什么问题】

Complete the requested producer-to-consumer comparison rather than stopping at
the first W/P2 defects.

【当时的 insight / hypothesis】

The large speed and memory gain could reflect deleted computation rather than an
equivalent refactor.

【具体做法】

Compared active observation/G and P1 call graphs to the complete local V120 tree,
including query factorization, candidate rematerialization and microgrid axes.

【观察到的结果】

- V120's `G block -> progressive updater` sequence rematerialized N=49 after G2;
  the current host mainly updated a public carrier.
- V120 used 24 horizon factual queries, four glimpse types, the full N=49
  posterior and 3x3 microgrid before organizing four action bases.
- The replacement used 96 horizon-by-basis queries and a global-K read.
- G3 parent update was about 50× smaller and P1 query variation used a different,
  much weaker structure.

【当时的解释】

The mainline had turned behavior-preserving refactoring into algorithmic
substitution. Passing shapes and faster throughput could not establish fidelity.

【后来怎么看（截至冻结终点）】

Retained. It became the central justification for Schema24's mechanical G/P1
restoration.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；Schema24 addressed it.

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3617-3627。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。

【关联】

Progressive G, V120 P1 and information-preserving refactors.

### CI-0297 — A pure issue ledger must remove disproved suspects as aggressively as it adds defects

【时间 / 版本】

Schema23 full-source audit closure, 2026-08-13.

【当时在解决什么问题】

The growing audit mixed confirmed defects, historical repairs and plausible but
unverified suspicions.

【当时的 insight / hypothesis】

A useful current issue ledger should contain only unresolved issues, label
evidence strength and delete both fixed and disproved entries.

【具体做法】

Immediately wrote confirmed P0/P1 items, kept incomplete boundaries explicitly
as “pending audit, not convicted,” then closed each against V120 and the active
consumer graph.

【观察到的结果】

S exact-null behavior, bottom state/history/transition aliases and a supposed
missing second generic P2 block were ruled out. P1 weak variation was not treated
as proof of a local disconnect before full structural comparison.

【当时的解释】

Retaining disproved suspects would repeatedly redirect later implementations to
already-cleared modules.

【后来怎么看（截至冻结终点）】

Retained as the operating rule for `CURRENT_MAINLINE_ISSUES.md` and later
information-conservation ledgers.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录；推断/未决
- 对话：visible messages 3613-3627。
- 段级工件：见 Segment 031 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`CURRENT_MAINLINE_ISSUES.md`。

【关联】

Evidence grading and issue-ledger hygiene.

## Segment 031 synthesis

Schema23 proved that exact Teacher scale and a live future-effect edge were not
enough: object identity remained weak and P2 null gradually removed the later
benefit. Same-iteration windows exposed an early-direct-path/late-structured-path
phase change. The complete local V120 tree then showed why: fake camera geometry,
changed optimizer lifecycle, a public-only substitute for progressive G and a
different P1 factorization. Just as importantly, the audit explicitly cleared S,
bottom aliases and a supposed missing P2 block. The next repair therefore had a
bounded mechanical scope rather than another broad redesign.

## Segment 032 — Schema24 mechanical restoration, early falsification and controlled source repairs

### Segment boundary

```text
Date:                 2026-08-13
Visible messages:     3639-3790
Starts with:          implementation of the faithful-main-path recovery plan
Ends with:            four source/log-confirmed repairs and three-round review
Primary historical artifacts:
  .audit/v120_exact_source_0b92d359/
  clearvla/mainline/model/v120_p1.py
  schema24_v120_fidelity_b8.log
  mainline_v120_contract_repair_b8.log
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0298 — Mechanical reuse was safer than an adapter that merely resembled V120

【时间 / 版本】

Schema24 implementation, 2026-08-13.

【当时在解决什么问题】

Restore progressive G, factual P1, the grounder objective and geometry without
inventing another approximation.

【当时的 insight / hypothesis】

The exact progressive organizer, address bank and P1 reader already existed in
the workspace. The lowest-risk repair was to make them active and remove the
substitute paths.

【具体做法】

- Ran each V120 grounding DiT block followed immediately by its progressive
  updater; G2 rematerialized N=49 and G3 exported facts.
- Restored the V120 24-query/four-glimpse/N=49/3x3 P1 reader.
- Restored one candidate K+null competition and dense reconstruction objective.
- Removed fake camera geometry and the cross-camera squared-distance penalty.
- Restored V120 decay grouping, decoder-local/global clipping and gradient stages.

【观察到的结果】

The principal source boundaries passed BF16 forward/backward and the final local
suite reported 146 passing tests. The model contained about `182,975,172` total
and `163,542,920` trainable parameters.

【当时的解释】

This was source extraction and call-graph restoration, not a new G/P1 design.

【后来怎么看（截至冻结终点）】

Partly retained. The mechanical classes were faithful, but subsequent logs found
additional adapter and target-reduction errors around them.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 3639-3655 and 3680-3681。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Progressive G, factual P1 and exact-source reuse.

### CI-0299 — `FactualPrecisionDock` was a typed cache boundary, not a replacement reader

【时间 / 版本】

Schema24 implementation review, 2026-08-13.

【当时在解决什么问题】

The new dock name raised the possibility that V120's main P1 computation had
again been crudely replaced.

【当时的 insight / hypothesis】

A boundary type is harmless only if it carries an already-computed value and
does not change producer inputs, consumer algebra, gradients or lifecycle.

【具体做法】

Traced the reader into the dock and dynamic P1, compared 3,458 non-empty source
lines with the V120 snapshot and tested chunked versus unchunked outputs and
parameter gradients.

【观察到的结果】

The dock contained only `[B,24,4,H] protected_detail`; the real computation
remained the V120 `LateRawDetailPolicyReader`, and dynamic P1 still ran at every
ODE node. Source and gradient equivalence checks passed.

【当时的解释】

The dock replaced the erroneous Schema23 global-K interface, not V120's factual
read.

【后来怎么看（截至冻结终点）】

Retained for this implementation boundary. Later concerns about static/dynamic
scale were distinct from whether the reader itself had been replaced.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已核实，不是粗暴替换。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 3663-3669 and 3673-3681。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Static factual cache and dynamic policy write.

### CI-0300 — Interrupted edits require an explicit half-state audit, not a fresh restart

【时间 / 版本】

Schema24 interrupted implementation, 2026-08-13.

【当时在解决什么问题】

Resume after interruption without either repeating completed work or overlooking
interfaces changed on only one side.

【当时的 insight / hypothesis】

The first action after interruption should be to map the current diff, failing
tests and every producer/consumer of changed types.

【具体做法】

- Found a diagnostic `future_address` reshape left behind after removing fake
  camera geometry and rebuilt its grid from real `object_to_chart` camera axes.
- Removed stale host/dock/reader references and corrected stale README identity.
- Resolved a trainable-but-unconsumed G3 generic route query by checking its true
  exported consumer instead of forcing a gradient.

【观察到的结果】

Core, full-suite, manifest, script, compile and diff checks closed without a
remaining half-wired interface. A missing recovery metric also became an explicit
audit result instead of an `IndexError`.

【当时的解释】

The interruption did not authorize skipping cleanup, and a passing core subset
was not evidence that historical interfaces had disappeared.

【后来怎么看（截至冻结终点）】

Retained as a process requirement and later strengthened into complete subsystem
familiarity before and after edits.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3656-3662 and 3671-3680。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`future_address`、`object_to_chart`。

【关联】

Half-implementation audits and optimizer ownership.

### CI-0301 — Schema23 long-run convergence masked a train/validation structural split

【时间 / 版本】

Schema23 epochs 1-7, reviewed 2026-08-13.

【当时在解决什么问题】

Use the longer predecessor log to decide what Schema24 actually needed to repair.

【当时的 insight / hypothesis】

Near-V120 aggregate RMSE could coexist with a weak structured top path and
validation-specific collapse.

【具体做法】

Audited six complete validations and the partial seventh epoch, separating
training medians from validation G/S/W/P metrics and checking exact Schema23
source rather than projecting Schema24 onto it.

【观察到的结果】

Training losses continued to improve, but validation stopped improving or
rebounded after epoch four. On the first diagnostic validation subset, S typed
innovation was about 1% of its training value, W interval change was below the
Teacher and P2 effect fell to roughly a quarter of training scale. P1 spatial
variation and gripper calibration also remained weak.

【当时的解释】

The direct action path could fit training while the structured future path did
not generalize or was poorly measured. Total RMSE alone could not prove recovery.

【后来怎么看（截至冻结终点）】

Partly retained. The validation collapse was real on the sampled batches, but a
later audit found that top diagnostics covered only the first four batches, so
it could not be generalized to the whole validation set.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；总体外推被收紧。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 3682-3689。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Validation sampling and train/validation phase gaps.

### CI-0302 — Schema24's large transport RMS was a reference-frame error, not observed motion

【时间 / 版本】

Schema24 epoch-one batches 20-2240, 2026-08-13.

【当时在解决什么问题】

Explain Teacher transport RMS around `0.25`, roughly 4.4× the early V120 value,
and weaker action optimization despite restored G/P1 machinery.

【当时的 insight / hypothesis】

If a motion target correlates with cross-camera coordinate disagreement rather
than flow, it is measuring a change of reference measure.

【具体做法】

Compared current and future coordinate aggregation, camera weights, learned-flow
RMS and four interval target shapes.

【观察到的结果】

Transport correlated about `0.922` with cross-camera coordinate differences and
about `-0.709` with learned-flow magnitude. The implementation independently
normalized per-camera coordinates and re-aggregated current/future states with
changing camera weights, producing apparent displacement.

【当时的解释】

The Teacher subtracted states expressed under different camera mixtures. P2 then
reasonably rejected corrupted geometric evidence; the W-to-action edge was
connected but fed the wrong target.

【后来怎么看（截至冻结终点）】

Retained as a confirmed target-algebra defect and fixed by computing same-camera
relative moments before object-level reduction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3694-3709 and 3735-3742。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Camera measure, coordinate frames and P2 rejection.

### CI-0303 — A lower flow loss can coexist with worse decoded action and weaker top information

【时间 / 版本】

Schema24 epoch-one aligned audit, 2026-08-13.

【当时在解决什么问题】

The batch-2240 point made Schema24 action-flow loss look better than V120, while
other indicators looked worse.

【当时的 insight / hypothesis】

One endpoint can be selected by noise and training phase. Windowed trajectories
must be checked before interpreting an apparent win.

【具体做法】

Recomputed all 112 shared epoch-one points in three windows: `20-400`,
`420-1100` and `1120-2240`.

【观察到的结果】

Schema24 was close earliest but worse in middle/late windows: action flow about
17.6%/19.3% worse, gripper 15.8%/26.8% worse and decoded action 19.8%/13.2%
worse. Flow magnitude fell while warp barely improved; reconstruction improved
while G3 correction and object identity weakened; stronger coordinate scores
produced weaker consequences.

【当时的解释】

The model found easier objective directions without preserving the structured
information required by decoded action. Low training loss was not synonymous
with a healthier causal path.

【后来怎么看（截至冻结终点）】

Retained. It became a recurring warning against judging top-path repairs by flow
loss or a single iteration.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3725-3734。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Shortcut detection and aligned-window audits.

### CI-0304 — Exact class bodies still failed when owners were frozen or public evidence duplicated

【时间 / 版本】

Schema24 post-epoch-one source audit, 2026-08-13.

【当时在解决什么问题】

Find remaining source-confirmed defects without rewriting already-cleared
P1/P2/P3/bottom computations.

【当时的 insight / hypothesis】

Copying an exact class body is insufficient if parameter ownership and adapter
inputs differ. Duplicate public evidence can also change a competition while all
expected tensors remain present.

【具体做法】

- Audited every frozen parameter with an active forward consumer against V120.
- Traced global-K candidate key/value construction.
- Rechecked transition, P1 dynamic replay and the non-mechanical S/W/P2/P3 paths.

【观察到的结果】

About `3,037,931` active observation/address parameters had been frozen. The
grounder also injected the same public chart key into every candidate, promoting
common identities. Transition still carried the dense spatial chart, and P1
dynamic replay, S/W core algebra, P2/P3 and bottom were cleared of an equivalent
disconnect.

【当时的解释】

The remaining defects were adapter/ownership errors around faithful bodies, not
evidence that downstream modules should be redesigned again.

【后来怎么看（截至冻结终点）】

Retained. Active owners were restored, only the genuinely unconsumed G3 generic
query remained frozen, and the duplicate public-key branch was removed.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3743-3761 and 3776-3787。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Parameter ownership, duplicate carriers and consumer-backed freezing.

### CI-0305 — Validation diagnostics and ablations must sample and name the same experiment

【时间 / 版本】

Schema24 epoch-one validation audit, 2026-08-13.

【当时在解决什么问题】

S/P1 validation diagnostics appeared to collapse, and current `full_updates`
appeared worse while V120 `full_capacity` improved action.

【当时的 insight / hypothesis】

These comparisons were invalid unless diagnostic selection, ablation semantics
and physical noise were identical.

【具体做法】

- Traced top/policy diagnostics to the first four validation batches.
- Mapped current ablation names to V120 behavior.
- Restored separate uniform budgets for structure, proposal and execution.
- Restored per-batch noise `37237 + batch_index`, shared across four ablations.

【观察到的结果】

Top diagnostics covered only about 2.23% of validation. `full_updates` actually
matched V120 `hard`, not `full_capacity`. Low S typed/P1 values were genuine for
those batches but could not be extrapolated to the full set.

【当时的解释】

Measurement identity was broken. Modifying S/P1 from this comparison would have
treated a sampling/name mismatch as a model defect.

【后来怎么看（截至冻结终点）】

Retained. Diagnostics were uniformly sampled and four true V120 execution
semantics were restored.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并修复。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预
- 对话：visible messages 3755-3767 and 3775-3790。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`full_updates`、`full_capacity`。

【关联】

Experiment identity, noise control and causal ablations.

### CI-0306 — Loss of trust converted closed-loop review into a release condition

【时间 / 版本】

Schema24 early failure and repair, 2026-08-13.

【当时在解决什么问题】

Repeated regressions after eight refactor iterations caused the user to question
whether the conversation and refactor should continue.

【当时的 insight / hypothesis】

V120 was not inherently unrecoverable. Exact source had repeatedly been treated
as conceptual reference while adapters, ownership, targets and lifecycle were
reimplemented. Passing shapes and gradients had replaced behavioral equivalence.

【具体做法】

The immediate proposal briefly swung to abandoning the reconstructed mainline
and running an untouched V120 worktree. The later, more useful effect was to
require a complete local snapshot, pure issue ledger, explicit no-change scope,
two-direction review and a fresh-run gate before claiming closure.

【观察到的结果】

The complete V120 snapshot was verified as 433/433 tracked files. The conversation
then returned from “discard everything” to four bounded source/log-confirmed
repairs, leaving S/P1/W observations explicitly unresolved.

【当时的解释】

The user's frustration was not technical evidence, but it correctly exposed that
the acceptance method had failed. Review rigor had to change, not merely the next
formula.

【后来怎么看（截至冻结终点）】

The proposal to discard all refactor work was not retained. The stricter
familiarity, provenance and closure requirements were retained and later
formalized in workspace instructions.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：过程要求保留；“全部丢弃”未采用。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；推断/未决
- 对话：visible messages 3710-3724 and 3768-3776。
- 段级工件：见 Segment 032 boundary 的 `Primary historical artifacts`。

【关联】

Mandatory subsystem familiarity and trust-preserving release discipline.

## Segment 032 synthesis

Schema24 correctly reactivated exact V120 G/P1 bodies and clarified that the
factual dock was only a cache boundary. Its early run nevertheless demonstrated
why source-body equality is weaker than graph equality: a camera-reference error
corrupted Teacher motion, active address parameters were frozen, public evidence
was duplicated into every grounder candidate and validation diagnostics compared
different samples and ablations. Windowed metrics showed that an easier flow
objective could coexist with worse decoded action and weaker structured
consequences. The response briefly overcorrected toward abandoning all refactor
work, but the durable correction was methodological: use the verified complete
snapshot, keep a pure unresolved-issue ledger, audit both directions and repair
only defects whose producer, consumer and observed consequence close.

## Segment 033 — Schema24 full-run diagnosis and the S ownership hypothesis

### Segment boundary

```text
Date:                 2026-08-14
Visible messages:     3791-3950
Starts with:          early log after the controlled Schema24 repairs
Ends with:            decision to concentrate subsequent repairs on S and necessary wiring
Primary historical artifacts:
  schema24_fidelity_fix_b8.log
  metrics.jsonl
  v120_long.log
  CURRENT_MAINLINE_REPAIR_PLAN.md
```

### CI-0307 — A rejected version is a diagnostic ancestor, not a performance target

【时间 / 版本】

Post-repair Schema24 audit, 2026-08-14.

【当时在解决什么问题】

The first response proposed comparing the new run with both V120 and the rejected
pre-repair Schema24.

【当时的 insight / hypothesis】

Performance conclusions require a declared primary anchor. A rejected ancestor
may identify whether a failure mode recurred, but it must not silently become the
target curve.

【具体做法】

After the user's correction, V120 became the sole behavior/performance baseline.
The older Schema24 was inspected only when the user later explicitly requested
its independent long-run audit.

【观察到的结果】

The new early run was close to V120 in action windows while still exhibiting
structural differences. The distinction prevented “better than a bad ancestor”
from being reported as recovery.

【当时的解释】

Comparison scope is part of experiment identity, not a presentation preference.

【后来怎么看（截至冻结终点）】

Retained and later refined into three anchors: V120 for regression behavior,
V25/V26 for achieved performance and separate structural-health criteria.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录；方案/设计记录
- 对话：visible messages 3791-3804。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Baseline discipline and multi-anchor evaluation.

### CI-0308 — G3 activity and temporal variation are descriptive, not monotonic quality scores

【时间 / 版本】

Schema24 epochs 1-3, 2026-08-14.

【当时在解决什么问题】

Interpret `G3 parent L1`, interval variation and temporal variation without
assuming lower or higher is universally better.

【当时的 insight / hypothesis】

These metrics measure activity or separation, not error. They become diagnostic
only jointly with provenance, object separation, targets and interventions.

【具体做法】

- Defined G3 parent L1 as mean absolute change from G2/binder ownership to the
  bounded G3-corrected posterior.
- Defined interval/temporal variation as variation across four intervals or 24
  temporal queries.
- Compared their values with object cosine, overlap, reconstruction and source
  perturbations.

【观察到的结果】

Near-zero variation can mean collapse, but large variation can be fixed identity
templates or noise. Smaller G3 correction can mean either a good parent or an
idle G3. No one of these quantities established health by itself.

【当时的解释】

A diagnostic becomes meaningful only when the changed axis is known and the
downstream consumer is sensitive to it.

【后来怎么看（截至冻结终点）】

Retained. Later audits repeatedly separated time-slot identity from conditioned
future content and global-K binder correction from progressive G3 updates.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；因果干预
- 对话：visible messages 3817-3819。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Metric semantics and source-aware diagnostics.

### CI-0309 — Training-flow and validation-RMSE “convergence speed” were not the same experiment

【时间 / 版本】

Schema24 epochs 1-4, 2026-08-14.

【当时在解决什么问题】

Validation appeared to approach V120 faster than the training flow objective,
which seemed like an unusual generalization pattern.

【当时的 insight / hypothesis】

The apparent paradox could be a protocol difference rather than validation
learning faster.

【具体做法】

Compared training and validation source paths, including random flow time, JEPA
cell mask, role/condition dropout, information-balanced training sampling and
five-step clean validation integration.

【观察到的结果】

Training measured one-step velocity error under random corruption and a sampler
enriched for event/high-motion windows. Validation measured an uncorrupted
five-step endpoint on the natural validation distribution. Their magnitudes and
rates were therefore not directly comparable.

【当时的解释】

The phenomenon did not imply that the model generalized before fitting training;
the two logs measured different inputs and prediction objects.

【后来怎么看（截至冻结终点）】

Retained as the explanation. Later comparisons continued to use training only
for optimization/path diagnosis and validation for deployed behavior.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 3842-3845 and 4119。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Train/validation protocol identity.

### CI-0310 — The S future target was made nearly public by using current support as future selection

【时间 / 版本】

Schema24 full-run source audit, 2026-08-14.

【当时在解决什么问题】

Explain why S typed innovation was moderately below V120 during training and
collapsed strongly on current validation diagnostics.

【当时的 insight / hypothesis】

The earlier split between loss support and selector validity had been applied
too broadly: the recognizer target also lost interval/object-specific future
selection.

【具体做法】

Compared the recognizer aggregation. V120 weighted semantic deltas by each
interval/object's future validity; the current version used
`current_loss_support.amax(camera)`, which was nearly one for every object and
identical across intervals.

【观察到的结果】

The target became close to a uniform K-object mean. Goal/history/public content
could fit it, while typed information remained only an optional correction. On
like-for-like training metrics, typed innovation was about 21-46% below V120;
the more dramatic validation gap could not be directly compared because V120
did not log validation S diagnostics.

【当时的解释】

Loss support should remain independent, but the future representation target
still needs future-specific selection. The public target was an important
amplifier, not proof that all action error came from S.

【后来怎么看（截至冻结终点）】

Partly retained. Later designs replaced the free recognizer target entirely, but
the warning against using a nearly constant mask to define future identity
remained central.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：确认的放大器；后续被更广的 S 监督重构吸收。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 3835-3841 and 3855-3856。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Loss support versus representation selection.

### CI-0311 — G3 anchor identity was erased by a shape-correct transition adapter

【时间 / 版本】

Schema24 full-run source audit, 2026-08-14.

【当时在解决什么问题】

Find why P1 saw detailed G3 facts while the later transition grew in magnitude
without comparable generalization.

【当时的 insight / hypothesis】

The 512-row transition contract preserved row count but not row provenance.

【具体做法】

Traced the final `[B,4,C,8,8,H]` G3 rollout into P1 and transition. P1 consumed
the real rollout; transition instead consumed the anchor mean
`public_scene_base`, copied it four times and added learned interval identities.

【观察到的结果】

The original `anchor x camera x cell` axis became
`interval-label x camera x cell` with identical factual content. Controlled-
transition RMS grew from roughly `1.37` to `2.72`, consistent with amplitude
compensation for missing identity. Shape validators did not detect the loss.

【当时的解释】

This was a clear mainline refactor regression: V120 passed the final rollout
directly to controlled dynamics.

【后来怎么看（截至冻结终点）】

Retained and mechanically fixed in Schema26 through a single G3 dock shared by
P1 and transition.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认并在 Schema26 修复。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 3850-3856, 3884 and 4011-4019。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`public_scene_base`。

【关联】

Axis provenance, transition inputs and shape-valid semantic regressions.

### CI-0312 — S typed collapse was active null rejection, not missing upstream evidence

【时间 / 版本】

Schema24 epochs 1-6, 2026-08-14.

【当时在解决什么问题】

Determine whether small `typed_innovation_rms` meant weak facts, sensible sample
selection or a shortcut.

【当时的 insight / hypothesis】

Raw value, route mass, sample/interval selectivity and final write must be viewed
together.

【具体做法】

Streamed the large JSONL and extracted raw typed value, innovation, null mass,
carrier ratio and route variation by epoch.

【观察到的结果】

Raw typed RMS remained about `0.55-0.64`, while innovation fell from `0.0598` to
about `0.0087`, null mass rose from `0.893` to `0.985`, carrier ratio fell near
1.5%, and sample/interval route variation nearly disappeared. This was a global
“take nothing” solution rather than conditional selection.

【当时的解释】

Goal/history/public paths could fit the public target and W could reread raw
typed facts, so permanently selecting null was a low-loss shortcut. The correct
repair was to remove substitutability, not impose a null quota or entropy loss.

【后来怎么看（截至冻结终点）】

Retained. Per-type zero-null and S-owned relevance were later introduced, but
subsequent versions showed that restoring route activity alone did not guarantee
healthy G/W identity or action gains.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已确认；后续结构修复部分吸收。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 3873-3888。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`typed_innovation_rms`。

【关联】

Optional innovation, null shortcuts and unique downstream ownership.

### CI-0313 — S had large parameter capacity but a very narrow effective typed outlet

【时间 / 版本】

Schema24 S capacity discussion, 2026-08-14.

【当时在解决什么问题】

Assess whether tangled S wiring mattered little because S itself was small.

【当时的 insight / hypothesis】

Parameter count and effective information bandwidth are different capacities.

【具体做法】

Counted active modules and traced dimensional reductions. The organizer had
about 29.4M parameters and CoarseAction about 14.8M, yet semantic, appearance
and geometry were each reduced to small route features, collapsed over K/type,
then passed through a roughly 66K shared router and a 98.5% null decision.

【观察到的结果】

More than 44M parameters repeatedly read and encoded conditions, while the
object/type-specific outlet was almost closed. Temporal expansion mainly
expanded already-public interval content.

【当时的解释】

The model did not need more S capacity. Existing capacity was poorly owned and
its informative axis was compressed before consumers could use it.

【后来怎么看（截至冻结终点）】

Retained as an early statement of the “capacity exists, usable bandwidth does
not” failure class that recurred in P1/P2/bottom audits.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录
- 对话：visible messages 3917-3926。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Information bottlenecks versus parameter capacity.

### CI-0314 — Sparse null writes are not a learned phase representation

【时间 / 版本】

Schema24 single-task interpretation, 2026-08-14.

【当时在解决什么问题】

The simple grab-pen task achieved strong action RMSE despite weak typed S,
raising the possibility that a few null patterns already encoded task phases.

【当时的 insight / hypothesis】

Public history, gripper state and action history can correlate strongly with
phase in one task, but null only suppresses optional content and carries no
positive phase semantics unless its posterior is explicitly consumed.

【具体做法】

Separated observable local state adaptation from explicit stage organization and
examined the available S inputs/outputs.

【观察到的结果】

History innovation was roughly `0.458` versus typed `0.0035` in epoch-six
validation. The model could infer progress from public history while using typed
residuals rarely. That did not demonstrate robust approach/contact/grasp/hold
state identity.

【当时的解释】

The simple task allowed a public main path plus sparse residuals. Multi-task,
ambiguous-object, recovery or non-monotonic trajectories would expose the absent
K/type/goal-conditioned stage structure.

【后来怎么看（截至冻结终点）】

Retained as a hypothesis with indirect evidence, not a measured phase-classifier
result. No phase loss or explicit phase machine was authorized.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：部分闭环
- 说明：保留为解释性假设。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 3927-3938。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Single-task shortcuts and future multi-task scalability.

### CI-0315 — Typed route features should select rich facts, not replace them

【时间 / 版本】

Schema24 S repair planning, 2026-08-14.

【当时在解决什么问题】

Semantic, appearance and geometry appeared information-poor after repeated
compression.

【当时的 insight / hypothesis】

Their small vectors were suitable as typed relevance keys, not as the full visual
content delivered downstream.

【具体做法】

Traced sources: semantic from projected DINO patches, appearance from raw/detail
features, and geometry from coordinate/flow/support/confidence features. The
proposed boundary retained `[interval,K,type]` relevance and let W/P use it to
read richer object content/detail.

【观察到的结果】

This avoided feeding full DINO content into S and avoided forcing a few 32-D
route vectors to carry all appearance or geometry.

【当时的解释】

S should organize and index conditionally relevant evidence; factual consumers
should retain ownership of rich values.

【后来怎么看（截至冻结终点）】

Retained in principle. Later implementations struggled with exactly where rich
values must remain protected and where policy residuals may enter.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；方案/设计记录
- 对话：visible messages 3937-3949。
- 段级工件：见 Segment 033 boundary 的 `Primary historical artifacts`。

【关联】

Selector/value separation and information-preserving docks.

## Segment 033 synthesis

The repaired Schema24 run eventually reached V120-level aggregate RMSE, yet its
source and JSONL showed two independent semantic faults: future S targets became
nearly public, and the real G3 anchor axis was replaced by a shape-correct public
copy before transition. S raw typed evidence survived, but a shared null globally
rejected it because public paths and duplicate W reads made it optional. This
clarified why strong action performance did not prove a healthy top path. It also
introduced several durable distinctions: descriptive variation is not a quality
score, training flow and deployment RMSE are different protocols, parameter
capacity is not usable information bandwidth, and null suppression is not phase
representation. The proposed response was to preserve public stateless control
while making K/type relevance a unique, optional selector of rich facts.

## Segment 034 — Schema25/26 ownership repairs and the broader Schema27 audit

### Segment boundary

```text
Date:                 2026-08-20
Visible messages:     3951-4168
Starts with:          complete eight-epoch Schema24 evidence and final S-only plan correction
Ends with:            Schema27 integration plus updated V25/V26 evidence
Primary historical artifacts:
  schema24_fidelity_fix_b8.log
  schema25_s_owned_typed_b8.log
  schema26_g3_s_boundary_b8.log
  v120_long.log
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0316 — A structural S defect was not proven to cause the epoch 7-8 rebound

【时间 / 版本】

Completed Schema24 eight-epoch run, reviewed 2026-08-20.

【当时在解决什么问题】

Correct the S repair plan using the full run rather than the epoch-six snapshot.

【当时的 insight / hypothesis】

A persistent architectural weakness and a temporally correlated regression are
different claims.

【具体做法】

Compared S public/typed, W intervals, P2/consequence and action components across
all eight epochs, especially E6-E8.

【观察到的结果】

Total RMSE rebounded about 2.6%, mainly in gripper and 5-24-step behavior. S typed
innovation rose from `0.0035` to roughly `0.0100/0.0086` rather than collapsing,
and W interval metrics stayed stable. P2/consequence effect increased, but this
was only a guardrail observation.

【当时的解释】

S ownership remained a confirmed defect, but the log did not establish it as the
direct cause of late rebound.

【后来怎么看（截至冻结终点）】

Retained. The episode became a model example of withdrawing an overstrong causal
claim while still repairing a source-proven defect.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：S defect retained; rebound attribution rejected.

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 3952-3960。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Correlation versus causal timing.

### CI-0317 — Schema25 made S the unique typed-relevance owner without forcing nonzero use

【时间 / 版本】

Schema25 implementation, 2026-08-20.

【当时在解决什么问题】

Close the shared-null and duplicate-read shortcut while preserving the public
stateless-control path and unchanged P/bottom machinery.

【当时的 insight / hypothesis】

The ownership switch had to be atomic: create a complete S-owned replacement and
remove CoarseAction/W's independent relevance decisions in the same change.

【具体做法】

- Split S public carrier from `[interval,K,type]` relevance.
- Gave semantic, appearance and geometry separate fixed-zero optional nulls.
- Made CoarseAction and W consume S-owned docks instead of rereading typed facts
  and running their own learned-null routers.
- Kept P1/P2/P3, transition and bottom internals unchanged.
- Added K/type perturbation, zero semantics, gradient and lifecycle tests.

【观察到的结果】

Duplicate readers/routers were removed, reducing about 12.73M parameters while
leaving bottom and exact P1 unchanged. Mainline plus log tests reported 153
passes. CPU BF16 passed; production CUDA memory remained unverified locally.

【当时的解释】

This repaired unique ownership and usable bandwidth without an entropy target,
quota, gain or new block.

【后来怎么看（截至冻结终点）】

Partly retained. Schema25 improved S/W object diagnostics but its early action
validation was worse; later source audits found that the wider G/W graph still
publicized identity.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；局部边界闭合，整体未闭合。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 3961-3981。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Atomic ownership migration and per-type zero-null.

### CI-0318 — Schema25 improved structure locally but inherited V120-relative slow early action learning

【时间 / 版本】

Schema25 batches 20-2180 and early validations, 2026-08-20.

【当时在解决什么问题】

Determine whether the S ownership repair itself caused the slower action-field
fit.

【当时的 insight / hypothesis】

Compare both to V120 and to the direct Schema24 parent; otherwise an inherited
gap can be misattributed to the new change.

【具体做法】

Used four training windows and later the `1920-2180` window, comparing action,
G/S/W/P and gradients to V120 and the parent.

【观察到的结果】

Late epoch-one physical flow was about 30% above V120 but about 2.6% better than
Schema24; decoded action was nearly equal to V120. G and W object cosine/overlap
became healthier than the parent. E1/E2 validation remained worse, while later
V25 reached `0.08980` at E3 and `0.07966` at E5, close to or better than the
corresponding anchors.

【当时的解释】

The S change was structurally active and not the source of the inherited early
optimization gap. Training flow also did not map one-to-one to five-step
validation RMSE.

【后来怎么看（截至冻结终点）】

Retained. V25 later became a performance anchor precisely because it combined
strong later validation with known structural debt.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 3987-4005, 4064-4069 and 4092-4099。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Parent-versus-behavior-anchor attribution.

### CI-0319 — Late RMSE rebound was localized to tail/gripper generalization, not global instability

【时间 / 版本】

V120 and Schema24 eight-epoch comparison, 2026-08-20.

【当时在解决什么问题】

Explain the best-to-final RMSE rebound shared by V120 and Schema24.

【当时的 insight / hypothesis】

The aggregate increase might come from a subset of horizons/components while
near action continued improving.

【具体做法】

Compared best and final full, first, tail, gripper, event and training-flow
metrics.

【观察到的结果】

Both runs improved training flow while validation rose about 2.6-2.7%. First-step
behavior continued improving; tail and gripper worsened. Schema24's gripper
rebound was larger. The dataset had only 63 training and 5 validation trajectories.

【当时的解释】

This was mild late overfitting/calibration drift concentrated in long-horizon and
sparse gripper behavior, not numerical divergence or a whole-model collapse.

【后来怎么看（截至冻结终点）】

Retained as a baseline defect shared by V120 and descendants. Later versions were
judged over full curves and best/final points, not a single epoch.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4004-4008。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Long-horizon generalization and event calibration.

### CI-0320 — Schema26 restored one factual axis and one S differential boundary

【时间 / 版本】

Schema26 implementation, 2026-08-20.

【当时在解决什么问题】

Repair the G3-to-transition regression and prevent the S typed route from being
injected twice without expanding into P/bottom redesign.

【当时的 insight / hypothesis】

A single typed dock should serve P1 and transition from the same final G3
rollout; a typed differential should influence W once and preserve exact zero.

【具体做法】

- Introduced one final-G3 rollout boundary shared by P1 and transition.
- Removed public-chart replication and learned pseudo-interval anchors.
- Composed S typed selection from common plus interval differential and kept one
  W entry.
- Removed unconsumed `future_address` online computation and no-op proposal
  ablation while preserving RNG draw order.
- Replaced ordinary LayerNorm on near-zero differential with a zero-preserving
  variance-floor normalization.

【观察到的结果】

Only 2,048 pseudo-anchor parameters were removed. Tests, static audits and
compiler checks passed. Schema25 exact resume was rejected.

【当时的解释】

The near-zero normalization correction was necessary because unit-normalizing a
tiny residual would recreate an earlier “small innovation amplification” defect.

【后来怎么看（截至冻结终点）】

Partly retained. The G3 factual-axis repair was durable; later logs showed that
global-K and W object identity could still be public even when the transition
axis was correct.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 4011-4063。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`future_address`。

【关联】

G3 dock, zero-preserving differentials and dead-interface cleanup.

### CI-0321 — Proposal-zero was a no-op by V120 design and should not be “restored”

【时间 / 版本】

Schema26 bounded audit, 2026-08-20.

【当时在解决什么问题】

The issue ledger listed proposal/seed equivalence as incomplete, and validation
still ran a proposal-zero ablation.

【当时的 insight / hypothesis】

Before restoring an apparently missing path, verify that the reference consumer
ever read it.

【具体做法】

Traced V120 generic P masks and bottom context. Both excluded proposal from the
active object mainline. The RNG draw was retained for reproducibility, while the
no-consumer metric and expensive ablation were removed.

【观察到的结果】

Proposal-zero was bit-exact because it changed no active consumer, not because
proposal robustness was perfect.

【当时的解释】

Restoring or optimizing this branch would have expanded scope around a historical
no-op.

【后来怎么看（截至冻结终点）】

Retained as a cleared non-problem and logging cleanup.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已排除并清理。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；过程/决策记录
- 对话：visible messages 4020-4048。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Dead-path audits and ablation semantics.

### CI-0322 — Schema26 proved that stronger transmission can carry lower-quality object information

【时间 / 版本】

Schema26 epochs 1-3, 2026-08-20.

【当时在解决什么问题】

Interpret lower P2 null and larger effect alongside worsening G/W object cosine.

【当时的 insight / hypothesis】

Path amplitude and path information quality must be evaluated separately.

【具体做法】

Aligned V120, Schema24, V25 and V26 at the same training age, comparing G/W
object and interval cosine, P2 null/effect and action flow.

【观察到的结果】

Schema26 action flow was competitive and P2 null fell to roughly `0.26` while
effect rose to about `0.068`. Yet G/W object cosine was systematically much
higher than V120; interval distinction was comparatively healthy. The apparent
G collapse was not monotonic—an early `0.83` peak later fell near `0.66`—but the
whole aligned window remained more public.

【当时的解释】

The downstream path was not starved. It more strongly consumed a poorly
identified K-object effect. The main fault was object-axis quality, not interval
axis or missing transmission.

【后来怎么看（截至冻结终点）】

Retained and repeatedly resurfaced in later W/P2 diagnoses.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4070-4099。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Amplitude versus information identity.

### CI-0323 — Historical replay must separate regression, inherited defect, amplified debt and correlation

【时间 / 版本】

Schema26 full-source replay preparation, 2026-08-20.

【当时在解决什么问题】

The goal was no longer merely to copy V120; some weaknesses visibly predated the
refactor.

【当时的 insight / hypothesis】

Each issue needed classification at the first semantic boundary where it arose:
new regression, inherited defect, inherited risk amplified past a threshold, or
correlation without sufficient causality.

【具体做法】

Established a fixed dataflow replay from Observation through G/S/W/P/transition
to bottom, using V120 source as behavior anchor and V24-V26 logs only to locate
when effects changed. The user required each finding to be written immediately.

【观察到的结果】

This prevented zero-time G/W modulation, 12 future supports, P1 docks and bottom
core from being repeatedly blamed, while exposing inherited long-horizon,
gripper, global-K and S/W weaknesses.

【当时的解释】

V120 could be both a regression anchor and an imperfect design ancestor.

【后来怎么看（截至冻结终点）】

Retained as the standard historical-audit method. Edge-by-edge note-taking was
explicitly adopted to survive conversation compaction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；过程/决策记录
- 对话：visible messages 4100-4113。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Historical logs as multiple views, not correctness targets.

### CI-0324 — The broader audit found several ownership and objective defects beyond S

【时间 / 版本】

Schema26 complete top-path audit, 2026-08-20.

【当时在解决什么问题】

Trace the entire active graph after repeated local fixes and determine which
remaining issues were source-provable.

【当时的 insight / hypothesis】

Healthy shapes, gradients and total RMSE could conceal free-coordinate targets,
unused candidacy fields, candidate-count priors and optimizer-owner mistakes.

【具体做法】

Audited every boundary and recorded findings as they appeared.

【观察到的结果】

Confirmed findings included:

- global-K dense mixture reconstruction did not identify K slots; typed evidence
  arrived only after binding;
- G `existence/null/allocation` had no W/P2 consumer while validity was nearly one;
- S public target was a jointly learned recognizer hidden and S/recognizer/W each
  had interval identities, allowing templates to satisfy supervision;
- W immediately summed semantic/appearance/geometry into one hidden;
- P1 dynamic self-write was roughly 8-10× static detail and the same fact was
  reused through several P lanes;
- one P2 null competed with 16 candidates, giving null only `1/17` mass at equal
  logits;
- execution controller was incorrectly included in decoder-local clipping;
- Teacher covariance omitted the null component's between-mean second moment;
- several logs conflated normalized tensors or two different “G3” boundaries.

【当时的解释】

These were distinct ownership, prior, target and observability faults. Not all
were proven main performance causes, but each had a source-level semantic error.

【后来怎么看（截至冻结终点）】

Partly retained. Schema27 attempted a coordinated repair; later experiments
showed that some fixes introduced new information bottlenecks, especially around
factual capacity and downstream compression.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；各子项已由 `CI-0335`～`CI-0362` 分别确认、修订或否定。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 4113-4133。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`existence/null/allocation`。

【关联】

Schema27 integrated repair and later capacity audits.

### CI-0325 — Schema27 tried to preserve type ownership through G/S/W/P rather than add pressure

【时间 / 版本】

Schema27 implementation, 2026-08-20.

【当时在解决什么问题】

Resolve the integrated G/S/W/P ownership faults without diversity loss, route
quotas or larger modules.

【当时的 insight / hypothesis】

Each boundary should carry uniquely owned, supervised and zero-defined
information; public state may modulate typed values but may not create them.

【具体做法】

- G used an independent DINO reconstruction target, typed pre-binding evidence
  and one physical K+null competition.
- S replaced a free recognizer coordinate with direct public/typed targets.
- W retained semantic/appearance/geometry sidecars to matching decoders; public
  W state could only multiplicatively modulate nonzero typed values.
- P2 corrected the candidate-count prior; P3 separated precision/effect/temporal
  sources and removed factual duplication.
- Controller clipping and Teacher null covariance were corrected.

【观察到的结果】

Static semantic tests, CPU BF16, lifecycle and 148 related tests passed. The
integrated model had about `167.1M` total and `150.7M` trainable parameters. No
new production result existed at this boundary.

【当时的解释】

The implementation attempted to make wrong shortcuts algebraically unavailable
rather than force desired metric amplitudes.

【后来怎么看（截至冻结终点）】

Only partly retained. Later V27/V28 evidence exposed additional publicization,
support and optional-capacity problems, showing that local ownership tests did
not yet close the full continuous network.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；实验后继续修订。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 4134-4161。
- 段级工件：见 Segment 034 boundary 的 `Primary historical artifacts`。

【关联】

Integrated ownership repair and subsequent information-conservation audits.

## Segment 034 synthesis

The full Schema24 run corrected a causal overclaim: S was structurally weak but
did not synchronously cause the late rebound. Schema25 then performed an atomic
S ownership migration, and Schema26 restored the real G3 transition axis plus a
zero-preserving differential. Their logs demonstrated a recurring distinction:
structured route activity can improve while action remains slow, and effect
amplitude can grow while K-object information becomes more public. The wider
source replay therefore stopped treating V120 as perfect and classified new
regressions separately from inherited or amplified debts. It uncovered free
targets, unused candidacy, candidate-count priors, clipping differences and
typed-value publicization. Schema27 attempted to repair that family as one
continuous ownership graph, but at this point its production behavior remained
unverified.

## Segment 035 — V25/V26/V27 aligned evidence and the first continuous ownership replay

### Segment boundary

```text
Date:                 2026-08-21
Visible messages:     4169-4270
Starts with:          complete comparison of the newest V25, V26 and V27 logs
Ends with:            source replay through Teacher loss support and the return to S
Primary historical artifacts:
  schema25_s_owned_typed_b8.log
  schema26_g3_s_boundary_b8.log
  schema27 integrated run log and metrics
  v120_long.log
  CURRENT_MAINLINE_ISSUES.md as it existed during the audit
```

### CI-0326 — V25 was the strongest completed action run, while V27 first improved object separation without faster action recovery

【时间 / 版本】

V25-V27 aligned review, 2026-08-21.

【当时在解决什么问题】

Determine whether the more explicit ownership structure had already surpassed
the older public-carrier implementation, using complete epochs and an aligned
training age rather than isolated early batches.

【当时的 insight / hypothesis】

Better object separation and larger structured-route activity do not imply a
faster reduction in deployed action error. Representation quality, route
amplitude and action utility are separate questions.

【具体做法】

Compared all completed validation epochs and then aligned the runs at
epoch 4/batch 2420 for the shared training metrics.

【观察到的结果】

- V25 completed eight epochs with physical RMSE
  `.10466 → .10385 → .08980 → .08274 → .07966 → .07724 → .07830 → .07887`.
- V26 reached `.10121 → .09431 → .08671 → .08526 → .08263 → .07893 → .08016`.
- V27's first three completed validations were
  `.09920 → .09941 → .09174`.
- V25's best `.07724` beat V120's best `.07931`; its final `.07887` also beat
  V120's final `.08145`.
- V27's W object cosine near `.460` was better separated than V25's roughly
  `.514` and V26's roughly `.559`, yet its action RMSE recovered more slowly.

【当时的解释】

V27 did not make every path weaker. It removed some easy common-mode behavior
before its more truthful boundaries had learned an equally effective low-frequency
action route.

【后来怎么看（截至冻结终点）】

Retained as evidence that health and action performance need distinct anchors.
Later versions repeatedly reproduced the same lag between internal separation
and deployed benefit.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4169-4174 and 4184-4187。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

V120 as regression anchor, V25/V26 as performance anchors, and later causal
consumption audits.

### CI-0327 — First/tail ratios and normalized/physical errors cannot be compared without their coordinate contract

【时间 / 版本】

V25 complete-run interpretation, 2026-08-21.

【当时在解决什么问题】

Explain why first normalized RMSE was near `.025` while tail normalized RMSE
was near `.249`, and whether the latter was directly worse than V120.

【当时的 insight / hypothesis】

The tail/first ratio is a temporal imbalance diagnostic, not a unit conversion.
Normalized and physical RMSE describe the same predictions in different,
per-dimension-scaled action coordinates.

【具体做法】

Verified the evaluation reduction and the action-normalizer inverse transform.

【观察到的结果】

At V25 epoch 8, `.248711/.0249356≈9.97`. This was not caused by the tail
containing more rows: both values were means. First-step error improved about
25%, whereas tail error slightly worsened. V25's normalized tail could not be
directly compared with V120's physical tail, and no single scalar converted the
whole tensor because each action dimension had its own scale.

【当时的解释】

The striking ratio primarily recorded near-action improvement with stagnant
far-action quality. It did not prove a tenfold physical execution error.

【后来怎么看（截至冻结终点）】

Retained as a metric-identity rule. Cross-run native/physical comparisons also
require the same normalizer fingerprint.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4178-4183。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.248711/.0249356≈9.97`。

【关联】

Horizon imbalance, normalizer provenance and full validation coverage.

### CI-0328 — The old common bottleneck carried useful low-frequency regularization, but its information-erasing implementation was not worth restoring

【时间 / 版本】

V25-V27 interpretation, 2026-08-21.

【当时在解决什么问题】

Decide whether the older public W/P bottleneck was partly necessary after the
cleaner V27 graph learned more slowly.

【当时的 insight / hypothesis】

There is a legitimate need for a stable shared future carrier in a simple
single-task dataset, but this does not justify averaging away K/type/interval
identity or allowing the common carrier to bypass the typed effect.

【具体做法】

Contrasted W/Teacher variation, W object cosine and action RMSE across V25 and
V27, then separated the bottleneck's function from its concrete algebra.

【观察到的结果】

V25 reached better early action quality despite more public W objects. The task
contained a strong common approach-localize-close trend, so forcing every useful
future signal through K, type, interval and null attribution increased
optimization difficulty.

【当时的解释】

The desirable design was framed as a protected low-frequency common future plus
identified K/type residual, not a return to the old early mean and bypass.

【后来怎么看（截至冻结终点）】

Partly retained. Later implementations repeatedly tried common/residual
decompositions, but also showed that merely naming two carriers does not close
their supervision, scale or terminal consumption.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；设计原则保留，具体方案继续演化。

【证据与锚点】

- 类型：源码事实；日志观测；方案/设计记录；推断/未决
- 对话：visible messages 4175-4177 and 4184-4187。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

Information conservation, public/typed ownership and later W common-mode audits.

### CI-0329 — Global-K reconstruction structurally rewarded suppressing null mass

【时间 / 版本】

Schema27 G source replay, 2026-08-21.

【当时在解决什么问题】

Explain why the global-K correction approached its bound while the posterior
barely changed and null mass collapsed.

【当时的 insight / hypothesis】

A null candidate is not semantically available if assigning mass to it creates
an unavoidable reconstruction hole.

【具体做法】

Traced the dense DINO reconstruction through K prototypes, the K+null posterior
and the exported object facts.

【观察到的结果】

The reconstruction had values only for the K prototypes and no protected/public
null reconstruction value. Any null mass therefore removed explanatory mass
from the chart. The sole G loss rewarded moving it back to K. This matched null
falling from about `.10` to `.003`, a bounded G3 compensation residual near
`.5`, and only about `1e-3` actual posterior change.

【当时的解释】

The correction spent amplitude on an object-vs-null common bias instead of
improving conditional K identity. Spatial separation could coexist with public
object content.

【后来怎么看（截至冻结终点）】

Retained as a confirmed source-level defect of that implementation. It must not
be confused with the local M-slot G2→G3 softmax, which had no null axis.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 4195 and 4208。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

Global-K identifiability, null semantics and reconstruction ownership.

### CI-0330 — G reconstruction could improve through a private decoder residual invisible to downstream consumers

【时间 / 版本】

Schema27 G source replay, 2026-08-21.

【当时在解决什么问题】

Check whether the independent DINO reconstruction actually supervised the
`ObjectFactSet.content` consumed by S/W/Teacher.

【当时的 insight / hypothesis】

A loss owner is not functionally closed when it can be satisfied by a private
value absent from the exported interface.

【具体做法】

Compared the reconstruction value with the object content exported from G.

【观察到的结果】

Reconstruction used `content + decode_content_residual(slots)`, whereas the
exported `ObjectFactSet.content` contained only `content`. The private residual
could therefore reduce G loss without improving the object value visible to any
downstream module.

【当时的解释】

This was a supervision-ownership gap. The log lacked the residual RMS, so it was
not assigned sole causal responsibility for object homogeneity.

【后来怎么看（截至冻结终点）】

Retained as an example of why target correctness is insufficient unless the
supervised value is the value actually exported.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：部分闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 4209 and 4238-4240。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`ObjectFactSet.content`。

【关联】

Private decoder shortcuts and producer-consumer identity.

### CI-0331 — The local G2→G3 common-residual diagnosis was withdrawn after axis inspection

【时间 / 版本】

Schema27 G correction audit, 2026-08-21.

【当时在解决什么问题】

Determine whether the tiny local G3 parent L1 represented the same null-bias
failure as the global-K binder.

【当时的 insight / hypothesis】

Identical metric names and visually similar common modes do not establish the
same algebra; the candidate axes must be checked first.

【具体做法】

Read the local owner update around its M-slot softmax and compared it with the
global K+null competition.

【观察到的结果】

Local G2→G3 had no null axis. A scalar common residual across M candidates
cancelled exactly under softmax. The tiny parent L1 could indicate a sharp G2
parent or a softmax-invariant proposal, but the available aggregate did not
distinguish them.

【当时的解释】

The earlier attempt to classify it as a second K-vs-null common-bias defect was
wrong and was removed from the problem ledger.

【后来怎么看（截至冻结终点）】

Retained as a documented self-correction and a warning against transferring a
failure across non-isomorphic axes.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已纠正；原故障归因否定。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录；推断/未决
- 对话：visible messages 4212-4213 and 4256-4258。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

Metric provenance and G3 boundary naming.

### CI-0332 — S supervision and W consumption lived in different learned coordinates, while the nominal zero-null default already wrote public typed value

【时间 / 版本】

Schema25+ S→W audit, 2026-08-21.

【当时在解决什么问题】

Explain why S typed losses improved but matching typed interval variation did
not survive the first W consumer.

【当时的 insight / hypothesis】

Supervising one decoded coordinate cannot establish information transfer through
another independent learned projection. Exact zero semantics also matter before
any learned specialization begins.

【具体做法】

Traced the S direct heads, the W-side consumption projection, public modulation,
per-type relevance and the policy-context K reduction.

【观察到的结果】

- S supervision and W consumption were separate learnable coordinates.
- Semantic interval distinction was already lost at W's first projection; the
  diagnostic was measured before public modulation, so this was not a logging
  artefact.
- `1+tanh(public)` could further attenuate a nonzero typed sidecar toward zero.
- The per-type “zero null” used a sigmoid, so score zero meant `0.5`, not neutral
  zero. With near-one validity, every K/type wrote half its typed value.
- `policy_interval_context` then averaged K, creating an object-public component
  and leaving the P2 intent score with interval identity but no K identity.

【当时的解释】

This was not merely a small S. The supervised quantity, consumed quantity and
claimed neutral value were algebraically inconsistent.

【后来怎么看（截至冻结终点）】

Retained. The sigmoid-at-zero issue was classified as Schema25+ ancestry, not a
Schema27-only regression.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；历史检索
- 对话：visible messages 4197-4200, 4210 and 4259-4260。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`policy_interval_context`。

【关联】

Typed information preservation, exact-zero contracts and conditional K reads.

### CI-0333 — W was mainly a public hidden modulator of S sidecars, not the supervised typed future owner its name suggested

【时间 / 版本】

Schema27 W source replay, 2026-08-21.

【当时在解决什么问题】

Explain why future losses could fall while W remained public and why far typed
fields did not appear to mature through W1/W2.

【当时的 insight / hypothesis】

A module is not the owner of a field merely because its hidden state gates that
field. Ownership requires the module's updated state to decode the supervised
and consumed value.

【具体做法】

Followed near and far sidecars through W1/W2, their public hidden states,
CoarseAction and the typed output decoders.

【观察到的结果】

W hidden mainly produced `1+tanh(hidden)` and multiplied the original S typed
sidecars. The far typed sidecar did not itself pass through the W1/W2 working
state. Public object/interval context entered W both directly and through a
CoarseAction token that reread public S/object/history, strengthening common
modulation.

【当时的解释】

W was a contextual gain on S values rather than a chronological object-future
carrier. This made W easy to bypass and explained why S/W field names overstated
functional ownership.

【后来怎么看（截至冻结终点）】

Retained as one of the recurring causes of W becoming an optional external
slot. Later rewrites changed the representation several times but repeatedly
had to re-audit this exact ownership boundary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：推断/未决
- 对话：visible messages 4207, 4218 and 4226。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

W common/residual design, far-horizon causality and duplicate public carriers.

### CI-0334 — P2's inherited semantic-first posterior prevented geometry/status from owning a different object or interval

【时间 / 版本】

V120 ancestry through Schema27, audited 2026-08-21.

【当时在解决什么问题】

Determine whether typed W fields were truly complementary at their action
consumer.

【当时的 insight / hypothesis】

Complementary values cannot remain complementary if one type first chooses the
single source posterior and all other types are only read at that selection.

【具体做法】

Traced P2 content, intent, coordinate and validity logits through interval/K
selection and subsequent value mixing.

【观察到的结果】

Semantic delta selected the unique interval/object posterior. Geometry and
status then reused it and could not select a different object or time. Predicted
visibility also reduced selector validity, allowing a disappearance/status
prediction to mask the candidate carrying its own evidence. Upstream, K had
already been averaged out of the S policy context.

【当时的解释】

This was an inherited P2 semantic-ownership defect, amplified by later high-null
behavior rather than newly invented by Schema27.

【后来怎么看（截至冻结终点）】

Retained. Many later P2 revisions were attempts to separate spatial/type
selection from the physical interval terminal without losing complementary
information.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；继承缺陷。

【证据与锚点】

- 类型：源码事实；历史检索
- 对话：visible messages 4200, 4210-4212。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

P2 semantic/geometry separation and selector-validity semantics.

### CI-0335 — Schema27's candidate-count correction created a persistent 1:1 candidate-set versus null prior

【时间 / 版本】

Schema27 early and epoch-four evidence, 2026-08-21.

【当时在解决什么问题】

Correct the earlier observation that one null competing with 16 ordinary
candidates received only `1/17` mass at equal logits.

【当时的 insight / hypothesis】

Cardinality correction is not neutral when applied to every candidate logit;
it changes the prior over the candidate set as a whole.

【具体做法】

Inspected the `-log(16)` term and aligned early P2 null/effect metrics with
Schema25/26.

【观察到的结果】

The correction made the 16-candidate collection and null each receive roughly
half the prior mass. At step 20, null rose from about `.065` to `.688` and W
effect fell from roughly `.0023` to `8.1e-5`; by epoch four null remained about
`.722`.

【当时的解释】

The term was not a transient optimizer stabilizer. It provided a persistent,
legal W-ignore route and was a definite Schema27 regression.

【后来怎么看（截至冻结终点）】

Retained as a general lesson: candidate cardinality, null semantics and optional
value ownership must be designed together, not corrected by an unexamined logit
offset.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；明确回归。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 4214 and 4240。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

P2 null shortcuts and later per-type/type-local terminal designs.

### CI-0336 — Schema27 removed factual precision and simultaneously added four learned pseudo-null exits

【时间 / 版本】

Schema27 P3→bottom audit, 2026-08-21.

【当时在解决什么问题】

Remove duplicated protected facts from P3 without discarding the unique
high-resolution role or making W optional.

【当时的 insight / hypothesis】

Subtracting a basis mean is safe only for a genuinely basis-specific signal.
An all-zero value with a learnable source identity remains a selectable routing
alternative rather than a harmless placeholder.

【具体做法】

Traced P1 static detail, dynamic P1, P3 precision/effect/temporal/factual lanes
and the bottom RoleDeltaAttnRes sources.

【观察到的结果】

- Precision was defined as `P1 - mean(action_basis)`. Because much of the
  high-resolution fact was shared across basis, precision RMS collapsed from
  about `.24-.27` to `.0055`.
- The fact still existed in protected base, so this was role-selective capacity
  loss rather than total information deletion.
- The factual lane was set to exact zero, yet its four basis rows retained
  independent learned source keys and competed alongside an explicit null,
  producing five selectable zero exits.
- Temporal became a pure `action × S-public` lane that did not read W
  consequence. Its RMS was about `.326`, near the `.35` bound, versus effect
  near `.082` and precision near `.0055`.
- Dynamic P1 remained active at about five times static detail, but static P1's
  24 queries, N=49 posterior, four glimpses and 3×3 microgrid were still present.

【当时的解释】

The attempted deduplication over-corrected: it preserved a protected fallback
while destroying precision's owner-specific view and created multiple legal
routes that could reject typed W value.

【后来怎么看（截至冻结终点）】

Retained as a clear regression family. The dynamic/static ratio remained a
suppression risk rather than proof that static detail itself had vanished.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；明确回归，尺度归因后来由 `CI-0381`、`CI-0384` 收紧。

【证据与锚点】

- 类型：源码事实；推断/未决
- 对话：visible messages 4201-4202, 4215-4217 and 4264-4268。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

P1 fact ownership, P3 lane semantics and null-capable bottom routing.

### CI-0337 — Teacher association was not accidentally frozen, but semantic future supervision was duplicated and status scale was easy

【时间 / 版本】

Schema27 Teacher/loss replay, 2026-08-21.

【当时在解决什么问题】

Test whether weak reliability came from an unintentionally frozen random
Teacher and whether the logged field losses represented independent targets.

【当时的 insight / hypothesis】

An alarming lifecycle hypothesis must be withdrawn when initialization and
intended no-grad ownership explain it. Conversely, algebraically equivalent
losses must not be counted as distinct semantic evidence.

【具体做法】

Checked projection initialization, `requires_grad`, no-grad/FP32 execution,
successor/delta identities, target RMS and the loss ledger.

【观察到的结果】

Teacher semantic/appearance keys were explicitly orthogonally initialized,
frozen low-rank coordinates; they were not trainable heads accidentally trapped
inside no-grad. Teacher retained `matched + p_null·current`, fixed interval
means, per-camera transport and no reliability shrinkage. However,
`successor=current+semantic_delta` held on both prediction and target sides, so
successor and semantic-delta losses were raw versus normalized/directional forms
of the same error. Target RMS was roughly semantic `.529`, transport `.330` and
status `.066`, making near-zero status predictions naturally cheap.

【当时的解释】

The suspected Teacher-freeze bug was rejected. Repeated semantic pressure and
unequal target scale were inherited objective biases, and low status loss did
not prove good state semantics.

【后来怎么看（截至冻结终点）】

Retained with the distinction between a deliberate fixed Teacher coordinate and
the still-open association-label identifiability problem.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：冻结错误已排除；重复监督风险保留。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 4203-4206, 4211, 4219 and 4269-4270。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`requires_grad`。

【关联】

Teacher isolation, target semantics and loss-budget interpretation.

### CI-0338 — Learned flow degraded monotonically across V25→V27, but the available gradients could not identify the responsible objective

【时间 / 版本】

V25-V27 aligned flow audit, 2026-08-21.

【当时在解决什么问题】

Determine whether worsening flow evidence was an independent compiler change or
a downstream-gradient side effect.

【当时的 insight / hypothesis】

A trend can be certain while its cause remains underdetermined. Aggregate owner
gradient norms cannot reveal conflicting objective directions.

【具体做法】

Compared the same epoch/batch and inspected active flow source, external weights
and all backward consumers.

【观察到的结果】

At E4/B2420, V25→V26→V27 changed as follows:

```text
warp                 .0911 → .0938 → .0983
native flow magnitude .692 → .610  → .467
confidence            .279 → .235  → .183
correlation entropy   .666 → .716  → .782
```

The active flow compiler and external loss weight had no matching structural
change. Flow received ordinary geometry plus action/G/P1 gradients, but logs
contained only their aggregate observation-owner norm, not objective-specific
gradients or angles.

【当时的解释】

The degradation was confirmed; attributing it to one top branch, changing
weights or detaching gradients was not justified without decomposition.

【后来怎么看（截至冻结终点）】

Retained as an unresolved source of geometry quality and as an example of the
right boundary between diagnosis and speculative repair.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：部分闭环
- 说明：现象保留；根因未定。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 4223-4225。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。

【关联】

Flow-DINO geometry, objective-specific gradient audits and P2 geometry.

### CI-0339 — The audit closed several false alarms and identified exactly which missing diagnostics blocked causal closure

【时间 / 版本】

Schema27 continuous replay, 2026-08-21.

【当时在解决什么问题】

Prevent a growing issue ledger from treating every large RMS, inherited
behavior or compact-log omission as a model fault.

【当时的 insight / hypothesis】

A complete audit must record exclusions as well as faults and distinguish an
unprinted computed scalar from evidence that was never computed.

【具体做法】

Reviewed lifecycle, optimizer/clipping, transition, history proposal, action
loss, reconstruction metric provenance and compact/JSONL coverage.

【观察到的结果】

- Static G/S/W/P1 were built once per observation; Teacher ran once per training
  batch and zero times in deployment; only dynamic P1/P2/P3/transition/bottom
  repeated over five Euler nodes plus endpoint heads.
- Controlled transition consumed protected consequence and its larger raw RMS
  was normalized at bottom ingress; V120 formal rollout/layer auxiliary weights
  were zero, so action-only training was not a new omission.
- `future_transition` really entered the representation group through the
  intent-structure weight; it was not audit-only.
- Main action flow used the V120-comparable objective; event balancing was an
  audit row.
- LR/warmup, optimizer ownership and V120 local/global clipping were not the
  source of slow recovery.
- History proposal exposed only observable history tokens and did not inject a
  future action proposal into W/P/bottom.
- Reconstruction `.2→.6` was a target-identity change: V25/26 reconstructed a
  self-consistent `content_slots` chart, whereas V27 used independent current
  DINO. The new target was more honest, but CI-0330's private residual shortcut
  remained.
- Train/validation mismatch involved both condition dropout and a train-only
  visual context mask; the compact log did not stratify action loss by either.
- Late rebound existed in V120, V25 and V26 and the 13-24-step rows still owned
  about 52.8% of action-loss mass, so it was not a tail-weight omission.
- Missing evidence included objective-specific flow gradients, condition-mask
  strata, G correction common/centered decomposition and P3 lane×basis route
  mass. Some S attention/state-change scalars existed in JSONL but were absent
  from nohup projection.
- A zero generic trajectory becoming a learned constant through affine bias was
  V120 ancestry and insufficient by itself to explain the V25-V27 regression.

【当时的解释】

The live fault family remained producer-consumer mismatch and optional-route
semantics rather than lifecycle, LR or a hidden Teacher leak.

【后来怎么看（截至冻结终点）】

Retained. Several later review failures came from declaring closure after shape,
gradient and lifecycle tests without these functional-consumption diagnostics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；多项疑点已排除，观测缺口仍开放。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 4228-4249 and 4250-4270。
- 段级工件：见 Segment 035 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`future_transition`、`content_slots`。

【关联】

Evidence discipline, metric provenance and producer→consumer / consumer→producer
review.

## Segment 035 synthesis

The first complete V25/V26/V27 replay separated action performance from
representation health. V25 remained the strongest completed run, while V27
already made W objects less common without converting that improvement into
faster action recovery. Source inspection then explained why local tests had
overstated the repair: global-K reconstruction punished null and had a private
decoder shortcut; S supervision and W consumption used different coordinates;
W was mostly a public modulator of S sidecars; and inherited P2 let semantic
choose the source for every type. Schema27 added two clearer regressions—a
candidate-set-versus-null 1:1 prior and a P3 redesign that erased the precision
owner while adding pseudo-null exits and a strong W-free temporal route. The
Teacher-freeze suspicion and several lifecycle, transition, LR and clipping
hypotheses were explicitly rejected. Learned flow degradation remained real but
causally unresolved. The durable lesson was that a healthy-looking field must be
the same tensor that is supervised, exported and necessarily consumed; neither
loss reduction nor a nonzero aggregate gradient proves that closure.

## Segment 036 — Schema27 audit closure, Schema28 ownership repair and Schema29 complementary P2

### Segment boundary

```text
Date:                 2026-08-21
Visible messages:     4271-4433
Starts with:          continuation of the full G→S→W→P→bottom source replay
Ends with:            confirmation that Schema29 repaired V28 loss support and P2 double-null
Primary historical artifacts:
  schema25_s_owned_typed_b8.log
  schema26_g3_s_boundary_b8.log
  Schema27 and Schema28 logs/metrics
  local exact V120 source snapshot
  CURRENT_MAINLINE_ISSUES.md as it existed during the audit
```

### CI-0340 — S's state/action history was a fabricated row pairing rather than a true temporal sequence

【时间 / 版本】

V120 ancestry through Schema27, discovered 2026-08-21.

【当时在解决什么问题】

Explain why S could report a large history innovation while temporal variation
and stage meaning remained weak.

【当时的 insight / hypothesis】

Concatenating tensors to the same row count does not create a valid temporal
history. Offsets, duplication and the time coordinate must describe the same
observations.

【具体做法】

Traced dataset offsets, S's state padding/current append, action history and the
time embedding row by row.

【观察到的结果】

State offsets were `[-8,-4,0]`, while action offsets were
`[-24,-16,-12,-8,-6,-4,-2,-1]`. S appended current state even though offset
zero was already present, repeated the earliest state to reach eight rows, and
formed approximately `[-8,-8,-8,-8,-8,-4,0,0]`. It then paired those rows with
the eight different action times and assigned a fabricated uniform
`linspace(-1,0)`. Only two of eight state-change rows could be nonzero.

【当时的解释】

The history writer could learn a dataset template but did not receive a coherent
state-action chronology. This was V120 ancestry, amplified when later P3
temporal routes became strong.

【后来怎么看（截至冻结终点）】

Retained as a confirmed interface defect. Schema28 attempted to replace it with
a real temporal history boundary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema28 中进入修复范围。

【证据与锚点】

- 类型：源码事实；历史检索
- 对话：visible messages 4284-4288。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

S temporal identity, state-change evidence and P3 temporal ownership.

### CI-0341 — Conditional typed reweighting did not enforce the claimed physical-support boundary

【时间 / 版本】

Schema27 G/Teacher replay, 2026-08-21.

【当时在解决什么问题】

Check whether semantic, appearance and geometry could only redistribute mass
inside the selected physical K support.

【当时的 insight / hypothesis】

Dividing a physical posterior back out of a conditional reweight can undo the
very support constraint that the interface claims to preserve. Softmax's
strictly positive tails are not a meaningful exclusion set.

【具体做法】

Expanded the typed posterior formula, inspected its denominators and traced the
three typed object coordinates into Teacher.

【观察到的结果】

The formula cancelled the physical prior and much of the physical posterior.
Because softmax supplied positive mass to almost every candidate, the assertion
that excluded candidates could not be revived was nearly vacuous. Teacher then
used semantic/appearance-rebound identities for future keys, physical-K content
for current/null reference, and a third geometry rebound for coordinates. Once
the typed posteriors diverged, one future delta combined non-identical object
coordinates.

【当时的解释】

Large typed values did not prove better physical-object information. The
re-binding changed identity after the physical K competition and then mixed
those identities inside the target.

【后来怎么看（截至冻结终点）】

Retained as an information-identity failure. Later G designs repeatedly moved
toward a single physical K identity with conditional typed evidence that could
not recreate excluded mass.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 4289-4293, 4305 and 4311。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

Global-K identity, Teacher target provenance and typed conditional reads.

### CI-0342 — `current_loss_support` was learned allocation mass, not physical observation support

【时间 / 版本】

Schema27/28 source audit, 2026-08-21.

【当时在解决什么问题】

Explain why typed S/W prediction amplitudes remained weak while public future
supervision learned easily.

【当时的 insight / hypothesis】

A learned assignment/confidence tensor must not decide which difficult rows
receive supervision. Loss support should describe observable physical validity,
not the model's present certainty.

【具体做法】

Traced `camera_evidence_mass`, `camera_validity` and
`current_loss_support` through top target construction and every S/W reduction.

【观察到的结果】

Schema28 connected `camera_evidence_mass` to `current_loss_support`. Across the
log, `4 × camera_evidence_mass + grounding_null_mass = 1` to about `2e-6`,
proving that it was global-K allocation quality, not independent support. It fell
from about `.22` to `.085` while physical camera validity remained near one.
Loss normalization prevented a simple global weight shrink, but selected a small
set of high-evidence rows and nearly excluded uncertain ones. An earlier
conditional average also made one support diagnostic almost identically one
whenever any tiny valid mass existed, obscuring this distinction.

【当时的解释】

The detached mask could not be intentionally reduced by gradient, but it formed
a self-reinforcing curriculum: uncertain object-camera rows received little
typed future supervision and therefore remained uncertain.

【后来怎么看（截至冻结终点）】

Retained as a definite Schema28 connection error. Schema29 changed loss support
to detached physical `camera_validity`; evidence mass remained audit-only.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；Schema28 接线故障确认，Schema29 源码修复的孤立行为收益未识别。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 4294, 4399-4400 and 4425-4433。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`current_loss_support`、`camera_evidence_mass`、`camera_validity`。

【关联】

Support/selector separation, self-masking and typed future supervision.

### CI-0343 — Several inherited constants survived nominal zero inputs and therefore could not establish information use

【时间 / 版本】

V120 ancestry audit, 2026-08-21.

【当时在解决什么问题】

Check whether zeroed goal or generic trajectory paths were true nulls and
whether their nonzero outputs proved language/geometry use.

【当时的 insight / hypothesis】

Zeroing values while retaining learned queries, masks, affine normalization and
bias can leave a learned dataset prior. A nonzero downstream tensor then proves
execution, not dependence on the removed information.

【具体做法】

Traced goal dropout and generic trajectory zeroing through their residual,
normalization and projection layers.

【观察到的结果】

With `goal_keep=0`, T5 values were zeroed but four learned goal queries, their
residual and FFN still produced a nonzero protected goal. The zero generic
trajectory similarly entered 24 geometry rows and could become a learned
constant through affine LayerNorm and biased Linear projections. Both behaviors
were inherited from V120; compact logs did not show how much bottom attention
they received.

【当时的解释】

These were identifiability debts, not sufficient explanations for V27 versus
V120 and not authorization for a broad rewrite.

【后来怎么看（截至冻结终点）】

Retained as audit cautions. Later zero/shuffle interventions had to operate at
the actual consumer boundary rather than infer use from a nominal input mask.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：部分闭环
- 说明：祖传风险；影响程度未定。

【证据与锚点】

- 类型：源码事实；因果干预；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 4301-4303。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

Language identifiability, bottom priors and exact-null semantics.

### CI-0344 — Cross-camera object coordinates incorrectly treated support width as evidence weight

【时间 / 版本】

Schema27 coordinate audit, 2026-08-21.

【当时在解决什么问题】

Verify how object coordinates and flow priors were aggregated across cameras.

【当时的 insight / hypothesis】

Geometric uncertainty/width and evidence probability are different quantities;
wider support must not automatically receive more authority.

【具体做法】

Followed camera posterior quality, `camera_validity`, support width and the
object coordinate reduction used by S/W/P2.

【观察到的结果】

The reduction used `camera_validity × camera_support`, while upstream
`camera_support` represented read radius/support width and was used by Teacher
as a geometry variance scale. A less precise, wider camera could therefore
receive greater coordinate weight.

【当时的解释】

The same tensor was assigned incompatible meanings at producer and consumer.
This was not the original V120 global-read aggregation semantics.

【后来怎么看（截至冻结终点）】

Retained as a camera-measure ownership defect that motivated later explicit
camera-aware P2 geometry.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 4313 and 4322-4324。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`camera_validity`、`camera_support`。

【关联】

Camera permutation, covariance-aware geometry and observable camera measure.

### CI-0345 — Fixed interval identity could dominate S variation while low-rank heads left most consumed hidden directions unconstrained

【时间 / 版本】

Schema27 S/W audit, 2026-08-21.

【当时在解决什么问题】

Determine whether nonzero raw interval variation demonstrated condition-dependent
stage understanding and whether falling S losses constrained the tensor W read.

【当时的 insight / hypothesis】

Interval query identity is an address, not observable intent value. A low-rank
readout can supervise a handful of directions while leaving a full hidden carrier
free to form a common shortcut.

【具体做法】

Subtracted cross-sample interval templates, compared train and validation
condition-centered variation, and inspected the ranks of public/action/status/
transport heads versus W's consumed hidden width.

【观察到的结果】

Only about one fifth of V27's raw S interval difference remained after removing
the common template. Later data corrected the claim that S was entirely fixed:
train public condition-centered variation was about `.143/.147`, but validation
was only `.0516`; semantic fell from about `.029/.026` in train to `.0105` in
validation. Public and CoarseAction supervision projected 512 hidden dimensions
to state/action dimensions, often only about seven directions; status/transport
heads projected route width to two. Public correlation also explicitly limited
differential range to `1-|common_score|`.

【当时的解释】

S learned some condition-dependent structure, but a fixed four-slot template
and generalization gap dominated validation. Healthy head losses did not close
the full hidden coordinate consumed by W.

【后来怎么看（截至冻结终点）】

Retained. Schema28 redirected main-path values to zero-preserving condition
innovation while keeping interval identity as query/address only.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；“完全固定模板”强结论已纠正。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 4307, 4309-4310, 4316, 4324 and 4331-4334。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

S common/typed decomposition, generalization and supervised-consumed coordinate
identity.

### CI-0346 — Diffuse Teacher matches could become full-strength public future targets despite low association confidence

【时间 / 版本】

Schema27 epoch-six Teacher audit, 2026-08-21.

【当时在解决什么问题】

Explain why W targets remained common even after K-object separation improved.

【当时的 insight / hypothesis】

Not masking loss by reliability avoids self-erasure, but a diffuse posterior
still changes the semantic target itself by averaging many patches. Neutral
fallback and target quality are distinct concerns.

【具体做法】

Compared Teacher null probability, association confidence and the exact
posterior-weighted successor construction.

【观察到的结果】

At epoch six Teacher null was about `.055`, implying matches for roughly 94.5%
of objects, while association confidence was only about `.298`. The broad
posterior was nevertheless used at full strength to average DINO content, and
semantic loss did not use reliability as a mask.

【当时的解释】

This did not mean low-confidence rows were ignored; they became low-specificity,
public future targets. The issue was target identity/quality, not simply target
amplitude.

【后来怎么看（截至冻结终点）】

Retained as a Teacher identifiability risk. Later plans deliberately avoided
reintroducing reliability shrinkage while seeking better association semantics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；根治需要独立可观测证据或更 faithful association，不用 reliability mask 掩盖。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4305-4306。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

Teacher dustbin, successor target quality and W publicization.

### CI-0347 — Schema28 attempted an end-to-end ownership repair without rebuilding P1 or bottom

【时间 / 版本】

Schema28 implementation, 2026-08-21.

【当时在解决什么问题】

Repair the confirmed Schema27 G/S/W/P defects as one continuous graph rather
than patching route amplitudes.

【当时的 insight / hypothesis】

Public scene content, K-object innovation, interval identity and typed future
values should each have one owner. The tensor supervised by an auxiliary head
should be the tensor consumed by the next stage.

【具体做法】

- Split G public scene from K-object innovation and restricted G3 correction to
  conditional K rather than object/null.
- Rebuilt S around real temporal history and zero-preserving condition
  innovation; fixed identity remained an address.
- Made semantic/appearance/geometry typed states pass through W1/W2 and attached
  supervision to the actual consumed field.
- Gave P2 each type its own posterior/null and removed duplicate typed ingress.
- Removed P3's zero factual pseudo-lane; precision read static P1 detail and
  temporal required S, consequence and action.
- Reused the single online CoarseAction output for both W and its target rather
  than forwarding the module twice.
- Fixed a half-written public/K interface, a typed double ingress and an FP32/
  BF16 public-content boundary found during reverse review.

【观察到的结果】

`176` tests passed, including CPU BF16, optimizer ownership, Teacher isolation,
permutation/zero semantics and five-step deployment. Parameter count was
`172,021,189 total / 155,626,868 trainable`; bottom remained `55,833,392`.
Static V120 P1 still performed 24 queries, N=49 and 3×3 values. The
`FactualPrecisionDock` cached that output; it did not replace the reader.

【当时的解释】

This was intended to close producer-consumer identity without adding blocks,
external losses or learned pressure.

【后来怎么看（截至冻结终点）】

Partly retained. Schema28 fixed real G/W ownership faults, but its early run
exposed two unclosed boundaries: learned allocation as loss support and the
inherited outer type softmax interacting with per-type nulls.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；两项关键回归由 Schema29 接续修复。

【证据与锚点】

- 类型：测试/运行观测；推断/未决
- 对话：visible messages 4335-4369。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`172,021,189 total / 155,626,868 trainable`。

【关联】

Schema28 implementation, mandatory double review and FactualPrecisionDock
semantics.

### CI-0348 — Schema28 repaired G/W object separation but worsened gripper/tail and weakened W consumption at P2

【时间 / 版本】

Schema28 epoch one and epoch-two mid-run, 2026-08-21.

【当时在解决什么问题】

Decide whether Schema28's structural improvements reached deployed action and
whether new risks justified stopping the run.

【当时的 insight / hypothesis】

The most informative causal clue can be an inverse trajectory: an upstream
field improves while its downstream selected value and gradient shrink.

【具体做法】

Aligned action, flow, G, S, W, P2/P3, gradients, runtime and the first validation
against V120 and Schema27.

【观察到的结果】

Epoch-one physical RMSE was `.10391` versus V120 `.09762`; arm was close
`.08407` versus `.08203`, while gripper was `.18215` versus `.16227` and tail
was `.32070`. Training flow and Flow-DINO diagnostics were not broken. G object
cosine fell from `1.0` toward `.47`, innovation cosine approached `.10` and
validation innovation cosine became negative; W object cosine improved to about
`.186` versus Schema27 `.461`. Yet as W interval variation rose
`.0377→.0443`, P2 null rose `.672→.682`, effect fell `.0376→.0309`, and
consequence ratio fell `.131→.112`. Runtime remained healthy near 11.8 GiB and
1.83 seconds/batch.

【当时的解释】

The principal bottleneck moved downstream to W→P2→P3. Better object and future
structure was being produced but increasingly discarded before action.

【后来怎么看（截至冻结终点）】

Retained. Later epoch-two evidence made the P2 and loss-support causes
structurally conclusive.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 4371-4385。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

Upstream improvement/downstream attenuation and action-consumption closure.

### CI-0349 — Schema28 fixed per-type posteriors but retained an outer type softmax, creating a second global null

【时间 / 版本】

Schema28 epoch-two diagnosis, 2026-08-21.

【当时在解决什么问题】

Determine whether status-type plus null was a reasonable no-effect decision or
an active W-ignore shortcut.

【当时的 insight / hypothesis】

Separating typed posteriors is only half a repair if an action-only outer mixer
then forces complementary fields back into mutual exclusion.

【具体做法】

Traced each type's K×interval+null read and the final
`softmax(type_query(action_query))`, then compared epoch-one and epoch-two
mass/value trajectories.

【观察到的结果】

From matched windows, future loss fell `.0637→.0282`, S typed context rose
`.0983→.1559`, and W interval variation rose `.0272→.0413`. Simultaneously,
semantic type mass fell `.164→.083`, geometry `.189→.064`, status rose
`.646→.851`, status null stayed near `.767`, P2 effect fell `.0346→.0264`,
and consequence interaction `.0063→.0040`. The type query saw only action, not
W evidence, per-type null or selected-value magnitude. Equal candidate logits
would give null only `1/17`, while observed status null was about `.767` under
mostly valid candidates.

【当时的解释】

Status could win the outer competition and then select its internal zero null,
discarding semantic/geometry even when they carried useful W values. The outer
softmax was inherited from V120; combining it with Schema28's three independent
nulls made the failure stronger. A previously recorded P2-type issue had been
closed after only the inner half was repaired.

【后来怎么看（截至冻结终点）】

Retained as a definite active defect and a review-process failure: checking
independent posteriors without following their terminal aggregation missed the
real information loss.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：Schema28 明确故障；Schema29 源码修复。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 4386-4405。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。

【关联】

Complementary type fusion, type-local null and premature issue closure.

### CI-0350 — A protected neutral anchor plus low-rank type-difference residual replaced mutual exclusion in Schema29

【时间 / 版本】

Schema29 design and implementation, 2026-08-21.

【当时在解决什么问题】

Remove P2's outer competition without freezing the model to a crude arithmetic
mean or introducing another learnable gate that could collapse.

【当时的 insight / hypothesis】

Complementary branches should retain their identities, contribute through a
noncompetitive protected base, and learn only bounded interactions that vanish
when the branches agree or are null.

【具体做法】

After literature/official-implementation review, used:

```text
base = (semantic + geometry + status) / 3
contrast = [semantic-geometry, geometry-status, status-semantic]
output = base + gamma * up(GELU(down(contrast)))
```

The contrast projections were bias-free and low-rank; LayerScale `gamma` began
at `1e-4`. Each type retained its own posterior and null. In the same narrow
release, S/W loss support switched from evidence allocation to detached physical
camera validity.

【观察到的结果】

The protected mean reproduced the old uniform-type behavior at initialization,
while the residual could learn type interaction without probability-mass
competition. All-null and equal-type inputs had exact-zero contrast correction.
`162` tests passed; BF16, optimizer ownership, manifest incompatibility and
scope containment were checked. Schema28 exact resume was rejected.

【当时的解释】

Fixed mean was treated as a stability anchor, not the model's whole fusion
capacity. The low-rank residual followed the common mature pattern of preserving
branch identity before fusion rather than learning a winner-take-all type gate.

【后来怎么看（截至冻结终点）】

Partly retained. Later Schema29 evidence showed that correcting support and
mutual exclusion was real, but the fixed `/3` protected mean introduced a new
scale loss and the low-rank contrast did not by itself preserve terminal typed
bandwidth. Schema30 repaired that local scale defect; still later versions
reworked the terminal again.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；protected complementarity retained, fixed `/3` realization rejected.

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 4405-4421 and 4427-4433。
- 段级工件：见 Segment 036 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`/3`。

【关联】

Schema29 complementary P2, support repair and low-rank residual fusion.

## Segment 036 synthesis

The Schema27 audit ended with a broader conclusion than its original three
regressions. S's temporal history was not a valid chronology; typed reweighting
could undo physical identity; camera width and evidence allocation were used as
if they were probability/support; fixed interval identity and low-rank heads
could make supervision look healthier than the full consumed hidden; and
diffuse Teacher matches could publicize future targets. Schema28 addressed many
of these ownership boundaries and demonstrably improved G and W object
separation without rebuilding V120 P1 or bottom. Its run then supplied the
critical inverse evidence: W structure improved while P2 effect and consequence
shrunk, gripper/tail worsened and null grew. Source replay found two exact
causes—learned allocation was misused as loss support, and the inherited outer
type softmax converted Schema28's three independent type-local nulls into a
second global W-ignore route. Schema29 repaired both in a narrow boundary,
using physical support and a protected mean plus low-rank type-difference
residual. The episode reinforced that a partial repair must be followed through
its final aggregation; otherwise a downstream operator can recreate the same
information loss under a cleaner intermediate interface.

## Segment 037 — Schema30-32, dustbin semantics and the information-capacity audit

### Segment boundary

```text
Date:                 2026-08-22
Visible messages:     4434-4736
Starts with:          complete Schema28 and partial Schema29 comparison
Ends with:            verified Schema32 early evidence and the qualified P2 factorization plan
Primary historical artifacts:
  complete Schema28 and Schema29 logs
  Schema30, Schema31 and Schema32 logs
  V120/V25/V26 historical logs and exact V120 snapshot
  CURRENT_MAINLINE_ISSUES.md
  CURRENT_INFORMATION_FLOW_ADAPTATION_ISSUES.md
```

### CI-0351 — Correct physical support exposed an older G bottleneck; it did not cause it

【时间 / 版本】

Schema28/29 full comparison, 2026-08-22.

【当时在解决什么问题】

Explain why Schema29's support repair coincided with stronger G/W publicization
and weaker transport/effect.

【当时的 insight / hypothesis】

A correct repair can expose a hidden producer defect by supervising rows that a
learned mask previously excluded. Reverting the repair would conceal the defect,
not solve it.

【具体做法】

Diffed Schema28→29, aligned the complete runs and checked V25 as a counterexample.

【观察到的结果】

Schema29 changed only loss support and P2 fusion in the relevant path; it did not
rewrite G/S/W. Support expanded from evidence mass near `.077` to physical
validity near `.854`. G object cosine rose toward `.844`, chart overlap toward
`.630`, and W interval/object structure weakened. Yet V25 had support near one
without the same severity, proving that full support was an exposer rather than
the root cause. The root producer defect was Schema28's simplified G
reconstruction, while Schema29 added a separate fixed-mean P2 bottleneck.

【当时的解释】

Schema29 stopped letting low-assignment objects evade supervision; the newly
visible failure belonged to the facts being supervised.

【后来怎么看（截至冻结终点）】

Retained. Physical support separation remained non-negotiable in later schemas.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；禁止以恢复 learned support 来掩盖。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 4434-4449。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Support semantics, G capacity and causal attribution of exposed debt.

### CI-0352 — Schema29's protected mean removed competition but diluted unequal complementary values and changed downstream initialization

【时间 / 版本】

Schema29 source replay, 2026-08-22.

【当时在解决什么问题】

Explain why fixing type competition restored some P2 effect but did not recover
V25/V120 action quality.

【当时的 insight / hypothesis】

A noncompetitive fusion can still be an information bottleneck if it permanently
averages branches with very different natural scales. Adding/removing modules
also invalidates a same-seed control when global RNG construction order changes.

【具体做法】

Measured per-type selected RMS, protected base, contrast residual and module
initialization order.

【观察到的结果】

Semantic/geometry/status RMS were roughly `.092/.031/.009`. Dividing their sum
by three yielded P2 effect near `.0336`; the `1e-4` low-rank contrast contributed
only about `.68%` of the base. This was better than Schema28's `.018`, but far
below V120's common `.08-.12`. New/deleted modules also changed global RNG
consumption, so downstream P1/P3/transition/bottom initializations were no longer
matched despite the same seed.

【当时的解释】

The outer-softmax shortcut was removed, but a weaker fixed attenuation replaced
it and the experiment included an initialization confound.

【后来怎么看（截至冻结终点）】

Retained. Later releases preserved old-module RNG or explicitly treated the
schema as a fresh, non-identical initialization.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema30 修复幅度代数，初始化控制成为持续审查项。

【证据与锚点】

- 类型：源码事实
- 对话：visible messages 4438-4444 and 4449。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.092/.031/.009`。

【关联】

Complementary fusion, scale preservation and controlled initialization.

### CI-0353 — Schema30 removed null-dependent reconstruction pressure and fixed P2 variance loss without adding capacity

【时间 / 版本】

Schema30 implementation, 2026-08-22.

【当时在解决什么问题】

Repair the two narrow Schema29 bottlenecks while preserving support, type-local
null and every unrelated module/state key.

【当时的 insight / hypothesis】

Learned null may govern optional online routing but must not turn off the
reconstruction pressure that identifies K content. Complementary zero-preserving
branches should use a variance-safe sum when their scales are not intended to be
averaged.

【具体做法】

- Reconstructed through G3 conditional-K ownership × local-M prior × observable
  validity, excluding learned-null value and denominator.
- Avoided re-normalizing an already conditional posterior and retained physical
  invalid-candidate fallback.
- Changed P2 base from `/3` to `sum/sqrt(3)` while preserving type posteriors,
  exact-zero null and the existing contrast residual.
- Kept parameters, 1429 state keys and initialization order unchanged.

【观察到的结果】

`211` mainline tests and `588` additional repository tests passed after excluding
one pre-existing legacy collection error. Parameters remained
`172,151,237 / 155,756,916`. Teacher, W, P1, P3, transition and bottom source
were bit-for-bit outside the two changed model files.

【当时的解释】

This was a narrow algebra repair, not restoration of V25's type softmax or
private reconstruction decoder.

【后来怎么看（截至冻结终点）】

Partly retained. The two local algebraic defects were fixed, but later evidence
showed that the G representation itself still lacked V120 reconstruction
capacity and W/P2 remained an end-to-end adaptation problem.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：局部修复保留；整体闭环未完成。

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 4450-4480。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`/3`、`sum/sqrt(3)`。
- 直接量值/表达式（卡内原文）：`172,151,237 / 155,756,916`。

【关联】

Conditional-K reconstruction, variance preservation and Schema30.

### CI-0354 — V30 improved object/typed structure but retained validation commonization and gripper weakness

【时间 / 版本】

Schema30 early-to-mid run, 2026-08-22.

【当时在解决什么问题】

Determine whether Schema30's two repairs restored V120 action behavior.

【当时的 insight / hypothesis】

Improved normalized, arm and object metrics can coexist with worse physical
gripper and a validation-only collapse of conditional temporal information.

【具体做法】

Compared epoch-two/three action bands and source metrics with V120, V25 and V29,
then traced the corresponding current implementation.

【观察到的结果】

At epoch two V30 physical RMSE was `.10523`; normalized/first/tail and arm were
better than V25/V29, but gripper was `.20389`, about 14% worse than V120
`.1784`. Validation W condition-centered variation was about `.0179` against
Teacher `.1367`, while the larger raw interval variation mostly contained a
fixed temporal template. G K innovation remained distinct and P2 effect returned
near V25 scale, so neither global-K collapse nor a broken P2 value path explained
the whole gap.

【当时的解释】

Schema30 fixed the intended local algebra but exposed S/W conditional
generalization and the gripper/precision path as the next weak boundary.

【后来怎么看（截至冻结终点）】

Retained directionally. Later complete V30 showed a best `.08556` at epoch six
and a nonmonotonic epoch-four rebound, still behind V120/V25.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 4481-4496 and 4560-4576。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Validation commonization, gripper/tail and full-epoch evidence.

### CI-0355 — Historical `p3_precision_rms` changed operands, invalidating the earlier direct “fine bandwidth loss” claim

【时间 / 版本】

V25→V30 P1/P3 replay, 2026-08-22.

【当时在解决什么问题】

Answer why precision appeared to lose the same fine-action bandwidth across
several revisions.

【当时的 insight / hypothesis】

Cross-version metrics with the same name cannot be compared until their source
tensor and ownership meaning are verified.

【具体做法】

Located the V25 commit and traced the real static P1 detail into protected base,
P3 and bottom through every intervening schema.

【观察到的结果】

The 24-query/N=49/3×3 reader was unchanged and its output still entered the
protected bottom path. V25 had duplicated the same full fact into factual,
precision and temporal lanes. V27 removed much of that duplication, so the
later `p3_precision_rms` measured optional precision innovation rather than the
same full fact. The drop `.260→.111` alone therefore did not prove that static
high-resolution information had disappeared.

【当时的解释】

The earlier repeated claim of definite fine-bandwidth loss was too strong. A
separate dynamic/static adoption test was needed.

【后来怎么看（截至冻结终点）】

Retained as a correction. Later evidence found a genuine reconstruction-capacity
loss elsewhere and kept P1 dynamic domination as a distinct risk.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：原直接归因已否定；P1 采用度风险保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 4497-4503。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`p3_precision_rms`。

【关联】

Metric provenance and protected versus optional precision.

### CI-0356 — The V25→V30 decline came from repeated local ownership cleanup without protecting the continuous transfer function

【时间 / 版本】

Schema25-30 retrospective, 2026-08-22.

【当时在解决什么问题】

Explain why many individually reasonable fixes yielded progressively weaker or
less stable action behavior.

【当时的 insight / hypothesis】

Correct ownership at each local interface is insufficient if every boundary
adds another average, null, selector or normalization. The full
`Teacher→S→W→P2→action` transfer function must be protected as one object.

【具体做法】

Replayed commits and archived V25-V30 curves, separating design intent, realized
connection and optimizer consequence.

【观察到的结果】

V25/V26 stayed near V120; persistent W publicization began in V27 and V28 was a
major turning point. Final/best RMSE and W/P2 structure changed approximately:

```text
V25 final/best .0789/.0772, W variation .099/.142 teacher, P2 effect .109
V26 final/best .0802/.0789, W variation .082/.130,         P2 effect .144
V27 final/best .0829/.0821, W variation .054/.097,         P2 effect .149
V28 final/best .0824/.0805, W variation .043/.060,         P2 effect .030
V29 final/best .0837/.0827, W variation .035/.055,         P2 effect .054
V30 partial    .0892/.0890, W variation .078/.124,         P2 effect .058
```

Schema27 simultaneously changed predictor, target scale and multiple
coordinates. Schema28 miswired support and type/null. Schema29/30 repaired the
terminal algebra after upstream bandwidth had already weakened. Proxy checks—
shape, pair cosine, lower future loss and nonzero aggregate gradients—had been
accepted without proving terminal action consumption.

【当时的解释】

Removing an impure but informative old route before establishing a legal
equal-bandwidth replacement created multiplicative attenuation across releases.

【后来怎么看（截至冻结终点）】

Retained as one of the central process insights. It led to the mandatory
producer→consumer and consumer→producer review rule.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；方案/设计记录
- 对话：visible messages 4502-4517 and 4577-4589。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Continuous-network review, action consumption and historical performance anchors.

### CI-0357 — “Clear” information ownership replaced “pure” path isolation as the design goal

【时间 / 版本】

Method correction during the V25-30 retrospective, 2026-08-22.

【当时在解决什么问题】

Correct the tendency to delete shared/common or multiply consumed paths merely
because they looked structurally impure.

【当时的 insight / hypothesis】

Clarity means knowing source, transformation, supervision and every consumer;
it does not require one path, one consumer or no fusion. A clean-looking
separated graph can be less intelligible if nulls and averages silently erase
its values.

【具体做法】

Reframed the review criteria around semantic identity and traceable consumption
rather than visual modular purity.

【观察到的结果】

Public/common information was recognized as legitimate when protected and
explicit. Repeated consumers were allowed when each transformation was named.
Semantic/geometry/status could fuse as complementary evidence. A shortcut was
defined by bypassing required evidence/supervision or exploiting leakage, not by
being short or shared.

【当时的解释】

The old path could be messy yet informative; later paths could be tidy yet
bandwidth-starved.

【后来怎么看（截至冻结终点）】

Retained as a general architecture principle.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 4588-4589。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Information conservation, common/residual carriers and review discipline.

### CI-0358 — Schema31 closed common/residual adoption but gave matching uncertainty an unjustified physical-disappearance meaning

【时间 / 版本】

Schema31 design, implementation and early run, 2026-08-22.

【当时在解决什么问题】

Prevent P2 null from discarding all of W and make diffuse Teacher targets,
S/W common-residual ownership and terminal consumption one closed chain.

【当时的 insight / hypothesis】

Common future should be protected while only optional interval residual may be
rejected. However, match failure and physical object disappearance are not the
same latent variable.

【具体做法】

- Added FP32/no-grad partial optimal transport with dustbin/identity fallback.
- Split S/W future into exact common plus zero-mean interval residual; removed
  duplicate CoarseAction typed ingress and repeated successor pressure.
- Made the supervised `FutureObjectDynamics` the value read by P2.
- Protected common from null and allowed null only for residual; predicted
  visibility no longer masked semantic/geometry candidates.
- Preserved prior effective raw/normalized/direction loss strength and module
  initialization RNG while removing dead parameters.

【观察到的结果】

`191` tests passed. At batch 2200 residual null was only `.028`, P2 effect about
`.171`, consequence ratio `.701`, and W was clearly consumed. Yet Teacher
dustbin was `.508`, best real minus dustbin `-1.92`, semantic delta `.190` and
status target `.517`. Status common `.272` exceeded semantic `.109` and geometry
`.041`. Action flow was still about 40% worse than V120 at that point.

【当时的解释】

The old W-ignore shortcut was closed, but the new protected route forcefully
injected a low-quality public status. Dustbin legitimately represented unknown
association for successor/transport; interpreting it as visibility/persistence
change turned epistemic uncertainty into physical disappearance.

【后来怎么看（截至冻结终点）】

Retained. Later Schema35 explicitly neutralized visibility/persistence when no
independent disappearance label existed.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：闭环修复部分保留；dustbin→status 语义被判为错误。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 4520-4559 and 4590-4616。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Teacher dustbin, common/residual ownership and protected P2 common.

### CI-0359 — Mature matching systems reinforced that unmatched, occluded and disappeared are distinct semantics

【时间 / 版本】

Schema31 dustbin design review, 2026-08-22.

【当时在解决什么问题】

Assess whether the partial-OT dustbin itself was wrong or only its downstream
interpretation.

【当时的 insight / hypothesis】

Partial matching needs a legal no-match state, but no-match should not be reused
as visibility/persistence supervision without independent observability.

【具体做法】

Compared the implementation with SuperGlue/LoFTR, DETR no-object and
TAPIR/CoTracker-style confidence/occlusion separation.

【观察到的结果】

The ClearVLA construction used K objects versus dense future patches, a fixed
zero dustbin score and 20 log-space Sinkhorn iterations, without ground-truth
correspondences. SuperGlue used a related dustbin but with supervised matches
and a learned scalar. Tracking systems explicitly separated match confidence,
occlusion and visibility. ClearVLA instead reused dustbin for identity fallback,
zero transport, uncertainty and negative visibility/persistence.

【当时的解释】

The Sinkhorn/dustbin mechanism was not inherently garbage; the semantic overload
was the defect.

【后来怎么看（截至冻结终点）】

Retained and incorporated into later Teacher contracts.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 4597-4605。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Association uncertainty versus physical state targets.

### CI-0360 — Deleting V120's small reconstruction decoders created a severe low-rank information contradiction, not a meaningful memory saving

【时间 / 版本】

Schema31 reconstruction audit, 2026-08-22.

【当时在解决什么问题】

Explain why global-K reconstruction remained around `.57`, more than twice the
V120 level, after null-independent pressure was restored.

【当时的 insight / hypothesis】

Removing a private-loss shortcut does not authorize removing the representation
capacity it carried. The legal replacement must export that capacity to the
canonical downstream fact.

【具体做法】

Compared current G with the exact V120 reconstruction and calculated parameter,
activation and optimizer-memory costs.

【观察到的结果】

V120 reconstructed with object content plus a slot content-residual decoder and
a coordinate decoder. Schema27 made the target an independent DINO chart;
Schema28 then removed both decoders and required `public mean + four constant
object residuals` to recover a dense `2×8×8×768` chart. The prediction was
restricted to a tiny affine span after local information had already been
pooled. The removed decoders had only about 405,504 parameters, around .77 MiB
BF16 weights and roughly 5-6 MiB including gradients/Adam; even major activation
buffers were only tens of MiB.

【当时的解释】

The implementation first discarded local content/coordinate information and
then penalized the model for failing to reconstruct it. This was an unauthorized
and practically pointless capacity cut made while trying to close a bypass.

【后来怎么看（截至冻结终点）】

Retained as a major error and review-policy change: save repeated computation,
materialized intermediates and dead paths—not coordinate bases, local detail or
consumer-visible capacity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema32 恢复 canonical 容量。

【证据与锚点】

- 类型：日志观测；测试/运行观测
- 对话：visible messages 4617-4629。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

G reconstruction, information bandwidth and resource-aware design.

### CI-0361 — The information-flow adaptation audit found few hard disconnects but several consumer-side capacity mismatches

【时间 / 版本】

Post-Schema31 whole-top audit, 2026-08-22.

【当时在解决什么问题】

Follow restored information downstream and find similar “preserved upstream,
compressed before consumer” defects.

【当时的 insight / hypothesis】

Most failures were not absent tensors. Rich information reached an adapter whose
operation—mean, last-row selection, multiplicative modulation or repeated target—
made it unusable in the next role.

【具体做法】

Created a deletable live issue ledger and audited mean/sum/expand/flatten across
G/S/W/P/transition/bottom axes.

【观察到的结果】

Confirmed issues included:

- G's dense target versus four-constant-vector capacity mismatch;
- W full object/action/goal condition only multiplicatively modulating a narrow
  typed state;
- W common bypassing W1/W2 while pre-W and post-W heads chased the same target,
  making identity correction a legal optimum;
- premature multi-camera coordinate averaging;
- fixed equal-logit consensus over correlated semantic/appearance and different
  geometry evidence;
- S→P1 reducing four goal rows to a mean and history to last-row replication;
- dynamic P1/self-write and biased zero-trajectory bottom paths as unproven
  adoption risks.

P1's high-resolution reader, transition's 512 rows, Teacher isolation and the
V120 bottom path remained continuous.

【当时的解释】

Fixing G alone would not make W consume richer facts. The canonical content,
W condition interaction and camera geometry had to be adapted together, while
unproven P1/bottom risks stayed out of scope.

【后来怎么看（截至冻结终点）】

Retained. It became the prototype of the mandatory subsystem-familiarity rule.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：核心 IF-01～04 进入 Schema32；其余继续保留为实验风险。

【证据与锚点】

- 类型：过程/决策记录；推断/未决
- 对话：visible messages 4630-4646。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Information-flow adaptation ledger and consumer capacity.

### CI-0362 — Schema32 restored canonical G capacity, W-owned common/residual processing and real camera geometry as one module package

【时间 / 版本】

Schema32 implementation, 2026-08-22.

【当时在解决什么问题】

Resolve the first four information-adaptation issues without expanding into
unproven P1, flow or bottom changes.

【当时的 insight / hypothesis】

New capacity is legitimate only when the reconstruction, online interface,
future target and action consumer share one canonical object. Full conditions
must create typed effects in a zero-preserving manner rather than merely scale
them.

【具体做法】

- Restored slot/position reconstruction capacity but wrote decoded slot residual
  into the sole exported object content rather than a loss-only private value.
- Made W1/W2 process both common and residual; added bias-free,
  variance-floor-normalized typed×full-condition interaction.
- Removed S's duplicate full FutureEffect target and retained its own observable
  state supervision.
- Kept camera hypotheses to P2 and used bounded log-mixture scoring; no support
  produced exact-zero geometry rather than uniform-camera fallback.
- Preserved old-module RNG despite new zero-initialized parameters and assigned
  every new parameter to an optimizer owner.

【观察到的结果】

`197` mainline tests passed. Parameters became
`169,622,469 / 153,228,148`, with 667,648 new owned parameters relative to the
immediate base after removed obsolete capacity. P1/P3/transition/bottom/flow were
not rewritten. G3 assignment and bottom neutral-trajectory adoption diagnostics
were added; uncertain IF-05-08 remained in the live ledger.

【当时的解释】

This was a coherent information-adaptation package, not a collection of gain
patches.

【后来怎么看（截至冻结终点）】

Partly retained. Correct early V32 evidence strongly supported G/S/W repairs but
showed that the now-distinct signed residuals cancelled inside P2's joint
interval×object terminal.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：核心修复保留；P2 消费需接续适配。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 4655-4681。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`169,622,469 / 153,228,148`。

【关联】

Canonical content, W condition interaction and camera-aware P2.

### CI-0363 — Later V30/V31 evidence showed stronger W common adoption without enough interval-specific action benefit

【时间 / 版本】

V30 seven epochs and V31 two epochs, 2026-08-22.

【当时在解决什么问题】

Check whether Schema31's closed common path solved the V30 late behavior and
whether Schema32 addressed the right defects.

【当时的 insight / hypothesis】

Restoring field amplitude can replace “W ignored” with “W common dominates.”
The next consumer must be evaluated on retained residual, not just total effect.

【具体做法】

Aligned complete V30, early V31, V25 and V120 and mapped every V32 source change
to a prior failure.

【观察到的结果】

V30 validation reached `.10485,.10523,.08902,.09711,.08917,.08556,.08678`;
the epoch-four rebound localized to tail/gripper as Teacher variation increased
while confidence fell and W object identity worsened. V31 increased W→P2 effect
about 3.6× but had weaker interval differentiation and no overall action gain.
V25 W2 adjacent cosine improved from `.942` to `.909`, close to V120, despite
very small S typed interval variation—evidence that V25 also relied on W identity,
typed action context and duplicate conditions.

【当时的解释】

Schema32 directly fixed canonical G content, the S/W identity shortcut, W-owned
common/residual and camera geometry. It only indirectly addressed the key
W→P2 residual attenuation, which required new-run evidence.

【后来怎么看（截至冻结终点）】

Retained; correct V32 early logs subsequently isolated P2 as the next bottleneck.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4682-4695。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Schema32 target mapping and W common/residual adoption.

### CI-0364 — Factorized interval→object P2 was a local code change with medium action and gradient consequences

【时间 / 版本】

P2 follow-up planning, 2026-08-22.

【当时在解决什么问题】

Prevent one joint softmax from averaging signed residuals across both interval
and object after W became more structured.

【当时的 insight / hypothesis】

P2 is a legitimate policy-conditioned soft-selection terminal. It should first
select the relevant future interval for an action time and then select an object
inside that interval, while protecting common effect and keeping types
complementary.

【具体做法】

Proposed `p(i,k|q)=p(i|q)·p(k|i,q)`, with an exact-zero residual null, no hard
phase, entropy target, gain or type softmax. Traced its unchanged ABI through
consequence, P3, bottom and backward.

【观察到的结果】

The code surface and compute cost were small and the `.35` contract bounded the
output, but every ODE step's effect and the action-loss gradient to S/W would
change. A mathematically exact factorization of the same logits would not solve
the cancellation; the new outer interval semantics had to be real.

【当时的解释】

It was classified as a local structural repair with medium behavioral impact,
not a trivial patch or a harmless checkpoint-compatible change.

【后来怎么看（截至冻结终点）】

Retained and later implemented in Schema33 after V32 showed W residual→P2
retention near 22%. Its directional factorization survived, while Schema33's
shared terminal-time authority was subsequently revised.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；Schema33 实施了 factorization，终端时间语义后来继续修订。

【证据与锚点】

- 类型：源码事实；日志观测；方案/设计记录；推断/未决
- 对话：visible messages 4694-4714。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

P2 terminal semantics and modular failure-family repairs.

### CI-0365 — Repairs should be grouped by a shared failure mode and closed dataflow, not forced into one experiment per line item

【时间 / 版本】

Planning-method correction, 2026-08-22.

【当时在解决什么问题】

Adapt “separate experiment, attributable, reversible” discipline to a large
continuous architecture whose local fixes depend on adjacent interfaces.

【当时的 insight / hypothesis】

The right experimental unit is a coherent failure family—trivial/default paths,
premature axis compression, bandwidth mismatch or consumer adaptation—not an
individual metric or function.

【具体做法】

Defined module packages with functional change, routing/normalization mechanism,
adjacent ABI adaptation, anti-shortcut boundary and per-boundary diagnostics.

【观察到的结果】

Related S interval-key, W residual and P2 terminal issues could be repaired
together because they shared an information-degradation chain. G3 correction,
P1 self-write and learned flow were kept separate when no common source-level
cause existed. A component need not independently improve RMSE if it closed a
confirmed necessary boundary without regression.

【当时的解释】

Attribution should come from stratified boundary evidence inside a coherent run,
not from breaking the continuous graph into nonfunctional micro-ablations.

【后来怎么看（截至冻结终点）】

Retained as the preferred planning granularity.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；推断/未决
- 对话：visible messages 4706-4714。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Modular repairs, failure families and continuous-network closure.

### CI-0366 — Run identity was falsified by active metric presence before interpreting the first alleged V32 log

【时间 / 版本】

First Schema32 launch attempt, 2026-08-22.

【当时在解决什么问题】

Analyze an early log labeled Schema32 after the new source was deployed.

【当时的 insight / hypothesis】

Output directory names are not run identity. Removed objectives and newly
required boundary metrics provide a stronger graph fingerprint than a label.

【具体做法】

Checked the log for deleted `object_intent_typed_future_field_loss` and required
canonical-slot/W-interaction/camera-mixture fields.

【观察到的结果】

All 59 records contained the removed typed-future loss and none of the new V32
fields. The user then confirmed the server had not pulled the new source. The run
was stopped and relaunched after checking commit `4ac7e54` in a new output
directory.

【当时的解释】

The first 1180 batches were a Schema31-like repeat, not evidence about Schema32.

【后来怎么看（截至冻结终点）】

Retained as an experiment-identity example.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已纠正；错误 run 不纳入 V32 结论。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 4716-4728。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`object_intent_typed_future_field_loss`、`4ac7e54`。

【关联】

Run context, source fingerprint and active-metric identity.

### CI-0367 — Correct V32 early evidence validated G/S/W repairs and isolated signed-residual cancellation in P2

【时间 / 版本】

Schema32 batch 2000-2320, 2026-08-22.

【当时在解决什么问题】

Verify that Schema32's canonical capacity and W adaptation worked, then locate
the remaining action-consumption loss.

【当时的 insight / hypothesis】

When upstream values become genuinely object/interval-specific, a joint soft
average that looked harmless on public values can begin cancelling useful signed
differences.

【具体做法】

Confirmed run identity and aligned G, S, W, P2, flow and action windows with
Schema31/V25/V120.

【观察到的结果】

G object cosine improved `.645→.498`, chart overlap `.338→.064`, innovation
cosine `.037→-.215`; canonical residual `.135` was 24% of public `.569`. S
residual/common improved from about 57% to 95%. W2 object cosine improved
`.555→.053`; common/residual states `.144/.134` and base interactions
`.049/.043` were balanced and active. Flow also improved. Yet W residual
`.0297` became P2 optional residual `.00666`, only about 22%; null was just
`.0235`, so rejection was not the cause. The joint interval×object softmax
averaged signed residuals that now differed more strongly.

Types also preferred different intervals—semantic/status mostly near and
geometry mostly far—risking a composite effect with no single temporal meaning.
This motivated a shared outer interval posterior and type-specific inner object
posteriors.

【当时的解释】

G/S/W were no longer the immediate bottleneck. Their newly recovered information
made the inherited P2 terminal mismatch observable.

【后来怎么看（截至冻结终点）】

Retained as the evidence that promoted factorized P2 into the next mainline
module package.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4725-4736。
- 段级工件：见 Segment 037 boundary 的 `Primary historical artifacts`。

【关联】

Schema32 validation and P2 factorized terminal.

## Segment 037 synthesis

Schema29's correct support repair exposed, rather than caused, a weakened G
producer, while its noncompetitive `/3` mean introduced a separate scale loss
and an initialization-control confound. Schema30 fixed those local algebraic
issues but did not restore the continuous future-to-action transfer. The
retrospective showed why: V25/V26 had retained strong, if messy, bandwidth;
V27-V30 repeatedly cleaned local ownership while adding averages, nulls and
selectors without proving terminal adoption. “Clear” therefore replaced “pure”
as the design goal. Schema31 closed common/residual adoption and proved that W
could no longer be ignored, but it also revealed a new semantic error: partial
OT dustbin encoded uncertain matching as physical disappearance and injected a
large public status consequence. A subsequent information-capacity audit found
that Schema28 had also deleted V120's inexpensive slot and coordinate
reconstruction capacity, forcing four constant vectors to reconstruct a dense
chart. Schema32 restored that capacity into canonical exported facts, let W own
both common and residual, and retained real camera geometry. The correct early
run strongly validated those changes and moved the remaining bottleneck to P2:
joint interval×object averaging cancelled the newly meaningful signed residuals.
The proposed factorized interval→object terminal was therefore a local code
change but a medium-impact behavioral repair, to be evaluated as part of a
coherent information-degradation module rather than a one-line patch.

## Segment 038 — Schema33-35, directional ownership, object-axis publicization and the scale-audit failure

### Segment boundary

```text
Date:                 2026-08-23
Visible messages:     4737-4980
Starts with:          mature Schema31/32 comparison and promotion of factorized P2
Ends with:            Schema35's 55x dynamic/static precision failure and the subsystem-familiarity rule
Primary historical artifacts:
  complete/extended Schema31 and Schema32 logs
  Schema33 owned-W2/factorized-P2 log
  Schema34 closed-loop S/W/P2 log
  early Schema35 log
  V120 and Schema25 behavior anchors
  00_CURRENT_ARCHITECTURE_CONTRACT.md
  CURRENT_MAINLINE_ISSUES.md
  CURRENT_INFORMATION_FLOW_ADAPTATION_ISSUES.md
```

### CI-0368 — Mature V32 evidence moved the dominant bottleneck upstream from P2 retention to W interval generation

【时间 / 版本】

Schema31/32 mature-run comparison, 2026-08-23.

【当时在解决什么问题】

Determine whether Schema32's early P2 residual loss persisted after training and
whether its improved G/S structure produced broad action recovery.

【当时的 insight / hypothesis】

A downstream retention ratio must be interpreted together with the amount and
structure of the signal the upstream producer emits. A P2 bottleneck can become
secondary if W itself learns mostly common future.

【具体做法】

Compared V31 and V32 at aligned validation epochs, decomposed Teacher/W/P2 into
common and interval residual, and separated physical, normalized, arm, gripper,
first and tail action metrics.

【观察到的结果】

At epoch 6, V32 physical RMSE was `.08015`, better than V31 `.08251` and close
to V120 `.08097`, but normalized RMSE, arm and long tail were worse than V31.
The gain was concentrated in gripper and object identity. G object cosine and
chart overlap improved, and S typed residual/common improved from roughly `.51`
to `.75`. Yet S condition-centered variation fell, W prediction variation fell
`.0426→.0254`, W adjacent cosine rose `.873→.946`, and residual loss worsened
while common loss improved. P2 retained roughly 59% of W residual late in the
run, close to V31's 63%, so the largest late bottleneck was no longer uniquely
at P2.

【当时的解释】

Schema32 improved which object was represented but still learned a mostly
common future. Factorized P2 remained structurally justified, but it could not
recover long horizon by itself unless W produced meaningful interval residual.

【后来怎么看（截至冻结终点）】

Retained. Later Schema33/34 evidence confirmed that producer structure and
consumer retention must be audited as one continuous transfer function.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4737-4741。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

V32 mature evidence, W interval bandwidth and P2 factorization scope.

### CI-0369 — Schema33 correctly made W ownership directional and factorized P2, but changed the terminal time semantics

【时间 / 版本】

Schema33 implementation, 2026-08-23.

【当时在解决什么问题】

Stop W2 near residual from contaminating protected common and prevent one
interval×object softmax from cancelling signed residual before action use.

【当时的 insight / hypothesis】

Common must not read optional near residual, while far residual should still
read near evidence. P2 should select a shared action-relevant interval and then
select an object separately for each typed value.

【具体做法】

Rewired W2 so near residual updated far residual but not common. Replaced three
independent interval×K competitions with one shared interval/null posterior and
per-type conditional object reads. Preserved exact-zero null and corrected its
neutral prior with fixed `-log(K)`. Added Jacobian, liveness, invalid-support and
typed-independence tests without adding parameters.

【观察到的结果】

All 180 regression tests passed and parameter count stayed
`169,622,469 / 153,228,148`. The required Jacobians held:
`d W2_common / d near_residual = 0` and
`d far_residual / d near_residual != 0`. The shared time posterior responded to
S public time evidence but was invariant to typed object/geometry values by
design.

【当时的解释】

This was an algebraic ownership repair rather than a capacity change. The
near→common behavior had been locally intentional as "refine common," but its
global effect was to broadcast interval-specific evidence to every interval.

【后来怎么看（截至冻结终点）】

Partially retained. The directional ownership principle was correct, but the
new shared time posterior gave a weak public-S key exclusive time authority and
therefore created a new behavior-level failure in the complete epoch-1 run.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；Schema33 终端时间语义后来被修正。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 4742-4771。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`d W2_common / d near_residual = 0`、`d far_residual / d near_residual != 0`。
- 直接量值/表达式（卡内原文）：`169,622,469 / 153,228,148`。

【关联】

W2 causal ownership, factorized P2 and null prior measure.

### CI-0370 — Early paired training equivalence did not license a behavior conclusion; validation and multiple anchors were required

【时间 / 版本】

Schema33 batch 680-1320 audit-method correction, 2026-08-23.

【当时在解决什么问题】

Decide whether Schema33 should continue after its first few hundred batches and
whether a direct-parent comparison was sufficient.

【当时的 insight / hypothesis】

Early paired training points can verify run health and that a mechanism is
active, but cannot establish generalization. Historical versions should be used
as selective counterfactual anchors, not as one giant undifferentiated table.

【具体做法】

Corrected an extraction bug that had matched `batch=680` from every epoch, used
strict `epoch=001`, compared paired windows rather than one batch, then adopted
V120, V25, the direct parent and current version as core anchors. Added only the
shortest ancestry version needed for each G/S/W/P2 question.

【观察到的结果】

Across 34-66 paired early points, action/native/decoded curves were almost
identical between Schema32 and 33. W interval variation increased and the new
near→far bridge and shared-time route were active. This supported continuing to
validation, but did not show an action benefit. The initial direct-parent-only
report was explicitly corrected after the user required V120/V25 context.

【当时的解释】

Comparison should answer whether a symptom is newly introduced, inherited or
amplified. It should not mechanically compare every metric from every version.

【后来怎么看（截至冻结终点）】

Retained. Schema33's first validation later demonstrated why the early
"not broken" result was only a run-health statement.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为实验比较方法。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；历史检索
- 对话：visible messages 4772-4795。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Aligned-window analysis, core anchors and run-health versus generalization.

### CI-0371 — Schema33's complete epoch 1 exposed a new long-horizon regression hidden by matching train curves

【时间 / 版本】

Schema33 epoch-1 validation, 2026-08-23.

【当时在解决什么问题】

Determine whether the factorized time route improved action behavior or merely
changed internal routing.

【当时的 insight / hypothesis】

A weak public scaffold can become a damaging bottleneck when promoted to
exclusive authority, even if all local ownership tests pass.

【具体做法】

Compared complete epoch-1 validation with Schema32, then traced residual/common
ratios and the exact inputs to P2's shared interval posterior.

【观察到的结果】

Schema33 versus Schema32 showed normalized RMSE `.29424` versus `.27684`,
physical `.10360` versus `.10122`, arm +7.2%, 5-12 +5.4%, 13-24 +6.6% and tail
+6.5%, while gripper was unchanged and aligned training action loss differed by
only about 0.12%. Teacher residual/common stayed near `.374`, but W fell to
`.230` and P2 to `.0621`. P2 retained only about 23.8% of W residual, and the
shared time route read only action time plus S public interval key. S public,
typed and temporal variation weakened; P2 intent score fell while coordinate
score grew. Null mass remained low, excluding a null shortcut.

【当时的解释】

Schema33 made a lightly supervised public-S scaffold the sole `when` owner. The
inherited diffuse Teacher, weak typed S and W residual then became a single
amplification chain. The result was a spatial/long-horizon generalization split,
not an optimization crash.

【后来怎么看（截至冻结终点）】

Retained. Schema34 reopened typed/W time evidence, confirming this source-level
diagnosis, although later object-axis problems remained.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema33 的新行为回归。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 4796-4806。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Weak-scaffold authority, W/P2 residual transfer and train-validation divergence.

### CI-0372 — History is a counterfactual instrument; the live issue ledger should contain only current problems

【时间 / 版本】

Issue-ledger and planning correction, 2026-08-23.

【当时在解决什么问题】

Record Schema33 findings without turning the current issue ledger into a version
chronology or forgetting inherited amplifiers.

【当时的 insight / hypothesis】

Historical runs are natural structural ablations: they reveal whether a
symptom was inherited, reduced, introduced or amplified. The live issue ledger,
however, should record only current source causes, impacts and closure criteria.

【具体做法】

Recorded current W→P2 compression, typed-value scale mismatch, diffuse Teacher,
S evidence asymmetry and metric-name debt; compressed ancestry into minimal
supporting evidence rather than a change log. Explicitly distinguished proven
amplifiers from merely competing paths.

【观察到的结果】

The ledger captured how diffuse Teacher, weak S typed residual, prior P2 loss,
typed scale mismatch and strong P1/bottom paths could combine with Schema33's
exclusive public time route. Resolved items were to be deleted rather than
retained as historical trophies.

【当时的解释】

An ancestry label is not a repair. History constrains the causal model; current
source determines what belongs in the active issue ledger.

【后来怎么看（截至冻结终点）】

Retained and later formalized in the workspace memory split between current
contract, current issues and auxiliary historical ledgers.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为文档边界。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；历史检索
- 对话：visible messages 4807-4827。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Current issue purity, historical counterfactuals and inheritance amplification.

### CI-0373 — Schema34 fixed a deterministic S-type wiring swap and restored plural time evidence

【时间 / 版本】

Schema34 source audit and implementation, 2026-08-23.

【当时在解决什么问题】

Close Schema33's weak exclusive time path and verify that S typed evidence was
mapped to the matching P2 owner.

【当时的 insight / hypothesis】

The producer and consumer type vocabularies were not positionally identical:
S exported semantic/appearance/geometry while P2 consumed
semantic/geometry/status. Positional indexing silently swapped geometry and
status conditioning.

【具体做法】

Made the mapping explicit as semantic←semantic, geometry←geometry and
status←appearance. Let shared time read public S, mapped typed S and supervised
W compatibility. Separated common and residual processing, retained near→far,
and added a zero-preserving one-sided value contract plus corrected routing
metrics.

【观察到的结果】

The swap was confirmed by source and type definitions rather than inferred from
logs. New tests established owner Jacobians, explicit mapping, time-route
responsiveness and a contract that could attenuate but not amplify small typed
values. The full local suite reported 185 passes with unchanged parameter count.

【当时的解释】

This was a deterministic connection error. Scale adaptation could not substitute
for mapping the correct semantic owner.

【后来怎么看（截至冻结终点）】

The explicit type map and plural time evidence were retained. The symmetric
common/residual isolation used in the same version was later found excessive.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；错接修复正确，所有权隔离需要后续修正。

【证据与锚点】

- 类型：源码事实；测试/运行观测；推断/未决
- 对话：visible messages 4828-4842。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Typed vocabulary mapping, S/W time evidence and zero-preserving adaptation.

### CI-0374 — Negative innovation-pair cosine was a centered-residual geometry, not automatically a collapse signal

【时间 / 版本】

Schema33 metric interpretation, 2026-08-23.

【当时在解决什么问题】

Explain why `innovation_pair_cosine` could be negative and decide whether more
negative was better.

【当时的 insight / hypothesis】

Cosine on centered object innovations has different semantics from cosine on
full object content.

【具体做法】

Verified that `content_innovation[k] = object_content[k] - public_content` and
that the metric averages pairwise cosines between those residuals.

【观察到的结果】

Different objects on opposite sides of the public mean naturally produce a
negative cosine. Four equal-norm, symmetric, zero-sum slots can yield the
reference value `-1/(K-1) = -1/3`. Values around `-.10` or `-.22` were therefore
not intrinsically abnormal. Near `+1` indicates publicization; near `-1` may
indicate two opposing clusters; any direction is uninformative if innovation
RMS is tiny.

【当时的解释】

The metric must be paired with full-content cosine and innovation magnitude.
"More negative" is not a monotonic quality target.

【后来怎么看（截至冻结终点）】

Retained as the correct metric semantics.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4843-4851。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`innovation_pair_cosine`。
- 直接量值/表达式（卡内原文）：`-1/(K-1) = -1/3`。

【关联】

Centered innovations, K-slot health and diagnostic interpretation.

### CI-0375 — Schema34 repaired time-axis consumption but exposed a time-versus-object axis tradeoff

【时间 / 版本】

Schema34 batch 800 through epoch-1 validation, 2026-08-23.

【当时在解决什么问题】

Verify whether plural P2 time evidence and the corrected type mapping produced
action gains without damaging G/W identity.

【当时的 insight / hypothesis】

A repair can improve one axis while making another cheaper to discard. Interval
variation is not sufficient evidence of healthy structured future if K-object
identity collapses.

【具体做法】

Tracked batches 200-2040, epoch-1 train and validation, and aligned V33/V32/V25/
V120 only where the metric semantics matched.

【观察到的结果】

By batch 2040, early G/S lag had mostly recovered and Schema34 improved training
action roughly 2-4% over Schema33, but remained behind V120 in near/mid horizon
and gripper. Epoch-1 physical RMSE `.10362` was essentially tied with Schema33
`.10360`; normalized metrics and gripper improved while arm worsened. Late in
the epoch G pair cosine rose `.494→.625`, chart overlap `.082→.244`, Teacher
dustbin `.470→.496`, and W object cosine `.095→.340`, while W interval
variation improved. Validation G pair reached `.637` and W pair `.307`.

【当时的解释】

Schema34 successfully made intervals different but allowed objects within an
interval to become similar. P2 then faithfully transmitted a more public W
signal. The version was locally effective, not a global recovery.

【后来怎么看（截至冻结终点）】

Retained. Later validation strengthened the train-validation object-axis
publicization diagnosis.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4852-4879。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Object-axis publicization, interval identity and upstream/downstream propagation.

### CI-0376 — Ownership isolation is directional; Schema34 cut the necessary common-to-residual anchor

【时间 / 版本】

Schema34 W source replay, 2026-08-23.

【当时在解决什么问题】

Explain why W object identity publicized after common/residual were made
"independent."

【当时的 insight / hypothesis】

Protecting common from residual does not imply mutual isolation. Residual needs
a read-only common anchor to remain object-conditioned.

【具体做法】

Compared the causal sequence used by Schema33 with Schema34's two independent W
block calls and ran a local Jacobian probe.

【观察到的结果】

In a causal `[common, interval...]` sequence, residual→common Jacobian was zero
while common→residual was nonzero (`2.6578`). Schema34 split the calls and made
both directions zero; its test even required this symmetric zero Jacobian.
Meanwhile W still mixed K internally and P2 could integrate K away before using
time compatibility.

【当时的解释】

The contract had confused non-contamination with disconnection. Removing the
common anchor made object-conditioned interval values harder to maintain and
turned public time structure into the cheaper solution.

【后来怎么看（截至冻结终点）】

Retained as a general ownership rule: arrows must be specified, not replaced by
blanket independence.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema34 的对称隔离判定失效。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 4880-4887。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Directional ownership, object anchoring and false isolation tests.

### CI-0377 — A full top-path review found several deterministic faults that local W/P2 review had missed

【时间 / 版本】

Schema34 whole-top source audit, 2026-08-23.

【当时在解决什么问题】

Move beyond the most visible W symptom and review the complete active top graph.

【当时的 insight / hypothesis】

Obvious failures identify where the main path is wrong; less obvious failures
are legal, cheap algebraic paths that quietly make the intended capability
optional.

【具体做法】

Traced Observation/Pre-G→G1/G2/G3→global-K→Teacher→S/CoarseAction→W1/W2→
static/dynamic P1→P2→consequence→P3→transition, including training targets,
deployment caching, ODE repetition and bottom handoff.

【观察到的结果】

The audit identified: W's four-interval recentering allowed far rows to rewrite
W1 near; Teacher interpreted association dustbin as physical disappearance;
dynamic P1 could generate a large "fact" from action query with zero detail;
P3 temporal read `P1+W` consequence and stayed nonzero when W was neutral; P3
precision ownership no longer matched static detail versus dynamic policy; S's
absolute future-state means favored public continuation; global-K repeated
content through typed identity votes; and camera-specific geometry was collapsed
too early. It also distinguished these source-proven faults from residual
cancellation that could still represent legitimate uncertainty.

【当时的解释】

Several apparently separate weak metrics belonged to ownership and target-
semantics clusters. Fixing only W amplitude would have treated their symptoms.

【后来怎么看（截至冻结终点）】

Retained as the source basis of the Schema35 plan. Subsequent early evidence
showed that even this broad review still omitted relative-scale competition at
the final precision consumer.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；审计范围正确，尺度审查不完整。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录
- 对话：visible messages 4888-4916。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Whole-top dataflow, factual/policy ownership, dustbin semantics and causal gauge.

### CI-0378 — Schema35 was designed as one causal-ownership repair family, not a collection of gains

【时间 / 版本】

Schema35 plan and implementation scope, 2026-08-23.

【当时在解决什么问题】

Repair G identity, Teacher semantics/camera geometry, S targets, W causality and
P factual-policy ownership without adding capacity or auxiliary pressure.

【当时的 insight / hypothesis】

The faults formed a continuous causal-ownership family and therefore required
adjacent ABI adaptation in one coherent implementation rather than isolated
amplitude patches.

【具体做法】

Planned single content K+null competition with conditional typed corrections;
dustbin-neutral status and identity-null geometry; adjacent S increments;
W1-near/W2-far causality; supervised typed W interaction; static fact versus
dynamic policy-precision split; camera-aware covariance P2; neutral W semantics;
and updated manifest/logging. Pre-G, static V120 P1 reader, transition and bottom
were held fixed.

【观察到的结果】

The plan was grounded in Schema34 RMSE `.1036/.0974`, public/typed/W P2 time
evidence around `.46/.18/.05`, Teacher dustbin near `.5`, dynamic P1 around
5.6x static detail and P3 precision below the V120 anchor. It explicitly forbade
new blocks, gains, quotas, hard gates and external losses.

【当时的解释】

The intended repair was semantic and causal, not an attempt to force target
amplitudes. Local health tests would still need to prove that each new owner was
both reachable and non-bypassed.

【后来怎么看（截至冻结终点）】

Partially retained. Several source faults were genuinely fixed, but the plan's
precision restoration criterion omitted an upper-scale and source-competition
check, allowing a severe new bypass.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；计划尺度验收不完整。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 4917-4919。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.46/.18/.05`。

【关联】

Schema35 causal-ownership package and non-expansion boundary.

### CI-0379 — Implementation review caught two real dead/biased branches, but still missed the final additive scale competition

【时间 / 版本】

Schema35 implementation and local review, 2026-08-23.

【当时在解决什么问题】

Close implementation-level dead paths in W and preserve dustbin uncertainty in
per-camera transport.

【当时的 insight / hypothesis】

Exact-zero semantics do not require consecutive zero-initialized layers, and
not treating dustbin as disappearance does not license renormalizing a tiny real
match into certain motion.

【具体做法】

Changed the typed×base W interaction from a zero-initialized two-layer dead
boundary to a `1e-3` near-identity initialization while retaining zero-input
zero output. Added dustbin as an identity-transport component in each observable
camera's full first/second moment and enforced FP32 PSD covariance. Fixed the
compact P1 logger to show the real 24-query/N49/3×3 path.

【观察到的结果】

Without the W change, zero interaction followed by zero effect heads denied
object content, transport prior and goal readers ordinary gradients for the
first updates. Without identity-null geometry, conditional renormalization
amplified a weak real association into strong motion. After repair, 220 local
regressions passed and low real-match mass continuously shrank transport toward
zero displacement.

【当时的解释】

These were meaningful implementation-review discoveries, but they did not test
the relative scale of static and dynamic sources after both reached P3.

【后来怎么看（截至冻结终点）】

The two fixes remained valid. The declared three-round review was nevertheless
not functionally complete because all rounds reused binary shape/zero/gradient
criteria and missed consumer-level dominance.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留修复；不保留“审查已完整”的结论。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录
- 对话：visible messages 4920-4930。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

W startup gradients, identity-null geometry and audit completeness.

### CI-0380 — Early Schema35 proved several target repairs active but showed W identity had not closed

【时间 / 版本】

Schema35 batch 1080, 2026-08-23.

【当时在解决什么问题】

Use the first live run to distinguish static implementation coverage from
behavioral closure.

【当时的 insight / hypothesis】

An implementation can correctly remove a target-semantic error and improve
downstream retention while still failing to preserve the identity structure of
the upstream value being retained.

【具体做法】

Checked run manifest and compared the batch 900-1080 window with Schema34,
focusing on Teacher status, S increment prediction, G separation, W object/time
structure and P2 residual retention.

【观察到的结果】

Teacher selector validity stayed at one, visibility/persistence losses approached
zero, S increment prediction/target RMS were `.232/.284`, G separation started
slightly better, and P2 W score rose `.073→.125`; retention improved
`.573→.746`, cancellation fell `.427→.254`, with null near `.036`. Yet W object
pair cosine worsened `.154→.470`, interval variation fell `.0222→.0159`, and W
residual fell `.0307→.0222`. With no validation yet, the dustbin/S/P2 repairs
were active but W object closure was not established.

【当时的解释】

Schema35 carried more W signal into P2, but that did not guarantee the signal
was object-specific. Retention and source health are separate requirements.

【后来怎么看（截至冻结终点）】

Retained as early diagnostic evidence; no claim about validation-side recovery
was permitted at this point.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 4931-4955。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Schema35 live verification, W publicization and retention-versus-content health.

### CI-0381 — Schema35 created a severe same-consumer dynamic/static precision scale competition

【时间 / 版本】

Schema35 early scale diagnosis, 2026-08-23.

【当时在解决什么问题】

Quantify whether dynamic P1 merely looked large in a different unit or actually
dominated static high-resolution facts at the same P3 consumer.

【当时的 insight / hypothesis】

Same-shaped sources sent through the same projection and added at one consumer
are directly comparable. A protected source can exist algebraically yet be
functionally absent because of scale dominance.

【具体做法】

Measured static factual base, dynamic policy residual, P3 precision, effect and
temporal at their actual addition point and checked the one-sided contract
saturation.

【观察到的结果】

Static fact RMS was `.0160`, dynamic residual `.8862`: about 55x amplitude and
roughly 3070x energy. P3 precision reached `.3407` under a `.35` bound, 97.4% of
the limit, versus effect `.0521` and temporal `.0942`. It was about 16x the
Schema34 same-stage precision. Dynamic P1 self-write also approached its `.5`
boundary. Because static and dynamic used the same precision projection before
addition, unit mismatch could not explain the ratio.

【当时的解释】

Dynamic policy residual could almost completely occupy the precision budget and
make static N49/3×3 facts irrelevant. This was an optimization shortcut and
bandwidth monopoly, not a numerical explosion.

【后来怎么看（截至冻结终点）】

Retained as a deterministic release-blocking error introduced by the Schema35
ownership adaptation. The source-level explanation continues in the next
segment.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；不需等待最终 RMSE 即可判定。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 4952-4960。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

P1 factual/policy ownership, P3 precision saturation and protected-base health.

### CI-0382 — Binary contract tests had repeatedly been mistaken for optimization closure

【时间 / 版本】

Review-method retrospective after Schema35, 2026-08-23.

【当时在解决什么问题】

Explain why repeated shape, gradient, zero-semantics and multi-round reviews had
still released obvious compression and scale errors.

【当时的 insight / hypothesis】

Functional closure requires favorable optimization geometry, not merely legal
interfaces. Repeating the same binary checks is not an independent review.

【具体做法】

Reconstructed the recurring cycle: remove one shortcut, transfer pressure to a
cheaper legal path, discover dominance only in logs, then patch the new symptom.
Converted the user's repeated closure requirement into explicit pre-run review
questions.

【观察到的结果】

Existing tests established shapes, nonzero gradients, detach boundaries, exact
zeros and parameter ownership, but did not ask whether one source could solve
the loss without required evidence, whether a protected base was numerically
submerged, where pressure moved after a repair, or whether every addition had
compatible units/RMS/Jacobian. Schema35's 55x ratio passed all prior binary
tests and exposed that omission.

【当时的解释】

The repeated failure was methodological: local legality had been reported as
end-to-end closure. The user explicitly required that a subsystem be fully
understood before any further modification.

【后来怎么看（截至冻结终点）】

Retained and promoted into the workspace-level mandatory subsystem-familiarity
rule: map all producers, transforms, consumers, losses, runtime calls, optimizer,
checkpoint, diagnostics and forward/backward scale boundaries before editing.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：已吸收为强制工作规则。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 4960-4979。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

Subsystem familiarity, optimization geometry, independent review and release gates.

### CI-0383 — The immediate call for wholesale rollback and the thirty-version count were both retracted as overcorrections

【时间 / 版本】

Schema35 failure response, 2026-08-23.

【当时在解决什么问题】

Decide whether repeated mainline errors justified discarding the accumulated
refactor and quantify how many historical versions were invalid.

【当时的 insight / hypothesis】

Behavior anchors and code rollback targets are different. Frustration-driven
global conclusions must be corrected with the same rigor as optimistic ones.

【具体做法】

Initially proposed returning to V120 and estimated at least 30 faulty releases
from memory. After the user's objections, explicitly retracted both statements
and separated ordinary research limitations from release-blocking static faults.

【观察到的结果】

The corrected position retained the independent mainline, V120 bottom,
flow-time/runtime, G1-G3, exact P1 reader, Teacher isolation, camera geometry,
S increments, W causality, P2 typed routing and the diagnostic/test system.
Only confirmed local algebraic regressions were to be replaced. V120 remained a
behavior and ancestry anchor, not a wholesale code rollback target. The earlier
"30 versions" figure had mixed known deficiencies with experiment-invalidating
errors and was withdrawn; the defensible conclusion was that release-level
static faults had recurred too often, not that nearly every version was useless.

【当时的解释】

The user's anger reflected repeated progress loss and tightened the review
standard, but it did not justify throwing away verified accumulated work or
inventing an unsupported count.

【后来怎么看（截至冻结终点）】

Retained as a self-correction chain. The later architecture replay applied this
distinction by separating inherited debt, behavior regressions and invalid
implementations instead of counting every version as one undifferentiated
failure.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：早期极端结论已撤回；修正后边界保留。

【证据与锚点】

- 类型：测试/运行观测；过程/决策记录；历史检索；方案/设计记录；推断/未决
- 对话：visible messages 4961-4979。
- 段级工件：见 Segment 038 boundary 的 `Primary historical artifacts`。

【关联】

V120 anchoring, evidence discipline, emotional events and conservative replay.

## Segment 038 synthesis

Mature Schema32 evidence showed that the first P2 bottleneck had partly recovered
while W itself increasingly produced common future; this qualified, rather than
cancelled, the need for a factorized terminal. Schema33 then correctly protected
W common from near residual and factorized interval/object routing, but promoted
a weak public-S key to exclusive time authority. Early paired training curves
looked healthy; the first validation exposed a clear long-horizon arm regression
and a two-stage W→P2 residual loss. This established the need for validation and
selective historical anchors rather than direct-parent or single-batch judgments.
Schema34 fixed a deterministic semantic/geometry/status versus
semantic/appearance/geometry wiring swap and reopened typed/W time evidence, but
made common/residual isolation symmetric. The resulting loss of the read-only
common→residual anchor improved interval distinction while G/Teacher/W object
identity publicized. A complete top-path audit then found additional source-level
faults in W far→near gauge, Teacher dustbin semantics, S absolute targets,
camera geometry and P1/P3 factual-policy ownership. Schema35 addressed many of
those faults and measurably improved P2 retention, but its dynamic policy
residual entered the same P3 precision consumer at 55x the static fact amplitude
and almost saturated the entire optional budget. That failure showed that shape,
zero, gradient and ownership tests are necessary but not sufficient: every
consumer must also be audited for units, relative RMS/Jacobian, saturation,
alternate solution paths and post-repair pressure transfer. The user's demand
that a subsystem be fully understood before modification was therefore promoted
to a workspace-level hard rule. The immediate suggestions to discard the whole
refactor and to label roughly thirty versions invalid were both later retracted
as unsupported overcorrections; V120 remained a behavior anchor, while verified
mainline improvements remained part of the code base.

## Segment 039 — Schema35-37 full-graph audit, information conservation and the W/P action-closure recurrence

### Segment boundary

```text
Date:                 2026-08-24
Visible messages:     4981-5214
Starts with:          source-level explanation of Schema35's 55x ratio and full-graph review reset
Ends with:            Schema37 early evidence, current problem set and corrected historical anchors
Primary historical artifacts:
  complete Schema35 failure log
  Schema36 P1/P2 closure log
  early Schema37 information-conservation log
  V120 exact source snapshot and long log
  Schema25/26 performance logs
  ClearVLA_schema25_35_architecture_audit_and_plan.md
  INFORMATION_CONSERVATION_LEDGER.md
  00_CURRENT_ARCHITECTURE_CONTRACT.md
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0384 — The 55x ratio was not itself newly generated; Schema35 changed its ownership and exposed an unstable reverse path

【时间 / 版本】

Schema35 full source/log replay, 2026-08-24.

【当时在解决什么问题】

Explain exactly why dynamic/static P1 differed by 55x and separate forward
scale mismatch from the epoch-3 non-finite failure.

【当时的 insight / hypothesis】

A large bounded forward residual may predate a refactor. The regression can be
where that signal is allowed to act, and an output RMS contract does not bound
the module's internal Jacobian.

【具体做法】

Verified run identity, traced static N49/3×3 detail, both dynamic P1 writes,
their P2/P3/transition/bottom consumers, and the action-loss gradient back
through the shared action canvas and observation encoder.

【观察到的结果】

Static detail was a `.25`-scaled innovation near `.016-.023`; dynamic P1 summed
two writes each allowed near `.5`, giving `.85-.89`. A similar ratio likely
existed in V120 but was not separately logged. Schema35's distinctive change was
to remove the dominant dynamic carrier from protected consequence, transition
and bottom base while leaving it in P2 query and direct optional P3 precision.
The run terminated at epoch 3 batch 1170. The first enumerated non-finite
parameter was flow `correlation_temperature_log`, but preceding raw gradients
grew `.277→1.84→4.65→831.6→5.17e6` along dynamic P1/shared canvas. P1 retained
about `.868` post-clip while W/P2/P3 were driven toward `1e-9`, so global clipping
had already turned a local instability into nearly whole-top training starvation.

【当时的解释】

The 55x value was a persistent ownership/scale signal, not the final-batch cause.
The P0 failure was the unbounded internal modulation/Jacobian plus inconsistent
consumer authority; the flow temperature was a first observed victim, not a
proven root cause.

【后来怎么看（截至冻结终点）】

This corrects the overstrong wording in CI-0381: Schema35 introduced the
same-consumer precision monopoly and ownership split, not necessarily the raw
dynamic/static ratio itself.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；对 CI-0381 的因果表述作后验收紧。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 4983-5003。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`correlation_temperature_log`。

【关联】

Dynamic P1, internal Jacobian, global clipping and first-nonfinite attribution.

### CI-0385 — A real whole-graph audit required independent online, supervision and optimization graphs

【时间 / 版本】

Schema35 audit-method reset, 2026-08-24.

【当时在解决什么问题】

Correct an audit that still expanded outward from the P1 failure rather than
examining the architecture without a presumed center.

【当时的 insight / hypothesis】

Coverage breadth is not a whole-graph review if the reasoning remains anchored
on one failure. Three complete graphs must close independently and then agree.

【具体做法】

Traversed: (1) data/cache→Observation→G→Teacher/S/W→P1/P2/P3→transition→active
V120 bottom→action/heads; (2) future supports→Teacher→every supervised tensor;
and (3) every loss→parameter owner→local/global clip→optimizer/checkpoint/log.
Included five Euler calls, endpoint heads, role-drop/action seed, history/coarse
proposal, CVAE/workspace and all bottom source competition.

【观察到的结果】

Dynamic P1 was absent not only from protected consequence but also transition,
the first bottom contract and event/motion evidence. Top language/S/W had no
direct CVAE-intent path; they depended on P3/transition. Optional P3 lanes were
further mixed by a nearly uniform 20-source AttnRes and fixed `.25`, unlike the
no-null protected reader. Large P1 gradients also shrank unrelated G/S/W/P
gradients under global clip. These cross-lifecycle and cross-consumer effects
were invisible in a P1-local review.

【当时的解释】

"The first broken node is P1" was a result of the completed graph, not the audit
premise. A nonzero P3 lane or gradient was insufficient evidence of action use.

【后来怎么看（截至冻结终点）】

Retained. This became the practical form of the mandatory subsystem-familiarity
rule and informed the later information-conservation audit.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为审查基线。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5004-5016。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Whole-graph audit, bottom consumer semantics and gradient-lifecycle coupling.

### CI-0386 — The external Schema25-35 audit was information-rich but had obsolete evidence and reversed priorities

【时间 / 版本】

Critical review of `ClearVLA_schema25_35_architecture_audit_and_plan.md`,
2026-08-24.

【当时在解决什么问题】

Decide which parts of an auxiliary audit should inform the next repair without
treating it as an active contract.

【当时的 insight / hypothesis】

An audit can be strong at locating anomalies and weak at causal priority or
implementation ordering. Facts, inferences and proposals must be separated.

【具体做法】

Rechecked its claims against current source, the complete Schema35 log and
active issue ledger rather than copying its older epoch-1 numbers.

【观察到的结果】

It correctly identified dynamic/static semantics, W-conditioned temporal,
shared P2 horizon, zero-target status still voting, fixed `/sqrt(3)` dilution,
over-hard W isolation and useful matched-noise/physical-band evaluations. Its
principle that ownership constrains source and write direction—not statistical
independence—was retained. But it ranked P2/W route redesign above dynamic P1,
while the complete log had already shown a P0 gradient failure and whole-top
clip starvation. Its log baseline and some "current" claims were obsolete.

【当时的解释】

The document was a research input, not executable authority. Dynamic P1 stability
and ownership had to precede optional route redesign.

【后来怎么看（截至冻结终点）】

Retained as an example of critical auxiliary-document use.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分吸收；原计划未直接执行。

【证据与锚点】

- 类型：日志观测；因果干预；过程/决策记录；推断/未决
- 对话：visible messages 5017-5024。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`ClearVLA_schema25_35_architecture_audit_and_plan.md`、`/sqrt(3)`。

【关联】

Auxiliary research boundary, evidence freshness and priority ordering.

### CI-0387 — Schema36 removed P1's duplicate precision authority and the false status consumer, but overcorrected by deleting valid precision bandwidth

【时间 / 版本】

Schema36 P1/P2 closure implementation, 2026-08-24.

【当时在解决什么问题】

Stop Schema35's P1 gradient monopoly and remove a P2 lane whose future value was
neutral but which still voted on time and diluted real values.

【当时的 insight / hypothesis】

Dynamic P1 should refine the P2 query, not independently create a second factual
base; semantic and geometry should each read matching evidence; neutral status
must not route actions.

【具体做法】

Removed dynamic P1 from direct P3 precision, kept static P1 as the factual
precision source, bounded P1 AdaLN smoothly at ±4, and added internal
modulation/QK/FFN diagnostics. Removed status from P2 route/value, gave semantic
and geometry separate interval/null posteriors with a shared public temporal
prior, and summed the two values without `/sqrt(3)` or type competition. Kept
neutral W status supervision diagnostic and recoverable under perturbation.

【观察到的结果】

At matched batch 1140, dynamic P1 fell `.8503→.1891`, self/FFN writes fell from
`.4865/.5000` to `.1072/.1194`, P3 precision input `.8503→.0138` and output
`.3400→.0519`. P1 raw gradients stayed around `.002-.010`; the old saturation
was absent. All 211 tests and lifecycle checks passed. Schema36 had
`169,475,012 / 153,080,691` parameters and retained the full ~55.8M bottom.

【当时的解释】

The P0 instability was addressed without a local clip. However, assigning
dynamic P1 exactly one query consumer turned an ownership correction into a
loss of valid action-conditioned precision bandwidth.

【后来怎么看（截至冻结终点）】

Partially retained. Status removal and matched typed routing remained useful;
query-only dynamic P1 was later classified as an overcorrection.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；P1 单出口设计后来被否定。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 5025-5050。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`/sqrt(3)`。
- 直接量值/表达式（卡内原文）：`169,475,012 / 153,080,691`。

【关联】

Schema36, dynamic P1 authority, neutral status and typed P2 routing.

### CI-0388 — Schema36's W looked catastrophically public early but partially recovered, requiring stage-aware qualification

【时间 / 版本】

Schema36 early and extended log review, 2026-08-24.

【当时在解决什么问题】

Measure how public W was relative to Teacher, Schema25 and V120, and determine
whether Schema36 had directly weakened W.

【当时的 insight / hypothesis】

Absolute producer/target health is more important than improvement over another
bad version, but early-stage severity must not be misreported as permanent.

【具体做法】

Compared common/residual, interval variation, adjacent cosine, object cosine and
dynamics gradients at matched early batches and later epochs. Checked the
Schema35→36 source diff.

【观察到的结果】

At batch 1140, W common/residual was `.0741/.0127` versus Teacher
`.1621/.0571`; interval variation retained about 22.5%, adjacent cosine was
`.970`, and common/residual ratio 5.8x versus Teacher 2.8x. This was roughly
3.5x less target-normalized variation than completed Schema25. Yet Schema36 did
not change W code, its 9.76M parameters, future loss or Teacher, and W gradient
remained `.120` versus Schema35 `.126`. Extended logs later showed epoch-4
common/residual reaching about 66%/62% of Teacher and adjacent cosine `.887`.
Epochs 1-3 physical RMSE `.09506/.09138/.08618` slightly beat V120, mostly via
arm, while gripper remained worse.

【当时的解释】

Zero-initialized W learned the easier common target first; constrained residual
was harder and action could rely on S/P1/bottom. Schema36 changed downstream
gradient geometry rather than shrinking W. The early problem was real, but the
later evidence downgraded "complete collapse" to delayed/insufficient residual
learning.

【后来怎么看（截至冻结终点）】

Retained with the later-stage correction. Comparisons to Schema36 must never be
used to declare another severely public W healthy.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留并已按后期日志收紧。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5044-5064 and 5146-5149。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.09506/.09138/.08618`。

【关联】

W publicization, stage-aware diagnosis and indirect gradient geometry.

### CI-0389 — W became optional through a sequence of locally reasonable safety and routing decisions

【时间 / 版本】

V120 through Schema36 ancestry replay, 2026-08-24.

【当时在解决什么问题】

Explain when and why W stopped being a functionally necessary future owner.

【当时的 insight / hypothesis】

Zero-preserving safety is not the same as permanent substitutability. A module
can begin as a bounded optional correction and later lose its only distinctive
responsibility to a parallel path.

【具体做法】

Traced the neutral-consequence rule, common/residual split, P2 time/object
factorization and public-S prior through V120, Schema31, 33, 35 and 36.

【观察到的结果】

V120 already allowed `W_effect=0` to fall back to current fact, reasonably
protecting early policy from bad future. But P2 still read W semantic delta for
time and W predicted full intervals. Schema31 introduced protected common plus
optional residual to stop whole-field null rejection. Schema33 split time from
object to solve interval×K cancellation, but assigned time largely to public S.
Schema35 made public S a protected time prior and imposed a harder residual
gauge. By Schema36, P1/current fact and S could support action while W supplied
only a small optional value.

【当时的解释】

Each motivation—safe cold start, protected common, factorized routing—addressed
a real local problem. Their composition converted temporary bounded influence
into permanent functional replacement.

【后来怎么看（截至冻结终点）】

Retained. The proper goal is not to force all action information through W, but
to ensure no same-semantics route can replace W's future-consequence role.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；安全语义保留，永久可绕过性不保留。

【证据与锚点】

- 类型：源码事实；历史检索
- 对话：visible messages 5065-5078。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

W optionality, neutral consequence and local-to-global composition failures.

### CI-0390 — Common/residual decomposition had diagnostic value but became an overconstrained optimization gauge

【时间 / 版本】

W design retrospective, 2026-08-24.

【当时在解决什么问题】

Decide whether common/residual itself was useful or should be removed after W
became weak.

【当时的 insight / hypothesis】

An exactly reconstructible decomposition can clarify ownership; treating both
components as separate primary carriers with a global gauge can delete legitimate
future freedom.

【具体做法】

Compared V120/Schema25 direct four-interval prediction with the later protected
common, four zero-mean residuals, separate processing and far-owned closure.

【观察到的结果】

Direct old W let W1 emit near and W2 emit far, then concatenated the full fields.
The later design forced residual zero mean while W2 alone had to repay the total
near+far gauge using its two far rows. Common was the easiest target and could
absorb signal; residual had harder causal and zero-sum constraints; P2 then had
to reconcile both carriers again.

【当时的解释】

Common/residual remained useful as a loss decomposition and diagnostic only if
the full four-interval field was losslessly reconstructible. The far-only gauge
was an unnecessary 25% interval-degree deletion, not causal protection.

【后来怎么看（截至冻结终点）】

Retained and implemented in Schema37 by deleting the far zero-mean gauge while
keeping derived common/residual views.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：分解保留；独立主路与 far-owned gauge 已否定。

【证据与锚点】

- 类型：推断/未决
- 对话：visible messages 5068-5078。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

W full-field freedom, near/far causality and diagnostic decomposition.

### CI-0391 — Information ownership had repeatedly been confused with exclusivity, causing at least six explicit losses

【时间 / 版本】

Schema36 information-conservation audit, 2026-08-24.

【当时在解决什么问题】

Quantify how much information had been lost while "clarifying semantics" and
separate necessary boundaries from overcorrections.

【当时的 insight / hypothesis】

Semantic correction is only lossless when the old information is reconstructible
or survives to a legitimate query-dependent terminal. Clear ownership does not
mean one carrier or one consumer.

【具体做法】

Classified every relevant boundary as exact reconstruction, conditional read,
hard loss, dead branch, supervision-only bypass or unresolved. Rewrote the
current contract as execution facts and created a separate information-
conservation ledger without modifying model code.

【观察到的结果】

Six explicit losses were recorded: appearance/status lost its action consumer;
dynamic P1 lost its P3 value consumer; semantic+geometry `2H→H` sum lost their
difference subspace; far gauge removed one of four interval degrees; camera
geometry value lost C before its terminal; S→P1 averaged K and then summed type.
Additional findings included public/typed S scale near 9.7x, W interval variation
near 40% of Teacher, direct P2 type cancellation near 30%, a reconstruction-only
G position head and bottom's joint lane/basis/null competition. Null mass was
only 3.65%, excluding whole-W rejection as the main issue.

【当时的解释】

Necessary rules remained: static fact cannot depend on noisy action, W2 cannot
rewrite near, dustbin is not disappearance, types retain distinct provenance,
and null cannot reject protected fact. The mistaken implementations deleted
legal outputs instead of constraining their write semantics.

【后来怎么看（截至冻结终点）】

Retained as the first explicit information-conservation inventory. It became the
source basis for Schema37.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；账本与“实际执行契约”已建立。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 5079-5104。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Information conservation, ownership versus exclusivity and contract truthfulness.

### CI-0392 — The repeated extreme corrections came from translating continuous tradeoffs into binary contracts

【时间 / 版本】

Method retrospective before Schema37, 2026-08-24.

【当时在解决什么问题】

Explain why successive versions alternated between excessive freedom and
excessive deletion.

【当时的 insight / hypothesis】

Easy-to-test algebraic properties can become harmful when treated as the design
objective rather than safeguards around a bandwidth-preserving representation.

【具体做法】

Mapped the recurring translations: "W matters"→delete other paths; "fact is
protected"→dynamic cannot reach P3; "types do not compete"→sum them; "far cannot
rewrite near"→global zero mean; "null is dangerous"→remove information.

【观察到的结果】

Bit-exact zeros, single consumers and zero Jacobians made attractive tests but
did not prove healthy bandwidth. Ownership was confused with monopoly, and
local failure modes triggered pendulum-like overcorrection. The key conceptual
correction was: a structural bottleneck organizes how information is used; it
must not become an information bottleneck.

【当时的解释】

Protected base, zero-centered bounded innovation, typed axes, multiple distinct
legal consumers and late query-dependent aggregation can coexist. Shortcut
control should constrain provenance and write semantics, not delete carriers.

【后来怎么看（截至冻结终点）】

Retained as a central design principle and reflected in the Schema37 plan.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：测试/运行观测
- 对话：visible messages 5084-5095。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Binary-contract pendulum, structural versus information bottlenecks.

### CI-0393 — Schema37 was an information-conservation implementation across S, W, P2/P3 and bottom

【时间 / 版本】

Schema37 planning and implementation, 2026-08-24.

【当时在解决什么问题】

Restore axes and legal consumers removed by Schema36 without rebuilding G,
Teacher association, static P1, transition or the active V120 bottom.

【当时的 insight / hypothesis】

Identity should survive until a real action-query consumer; future latent should
retain K/interval/camera; basis routing should operate after source ownership is
formed rather than fuse different information sources.

【具体做法】

S preserved public/goal/history/type with conditional-K reads; W removed far
gauge, kept W2 near-read/far-write and used appearance as a zero-preserving
semantic conditioner; P2 kept K×C to the geometry posterior and semantic versus
geometry sidecars; P3 exposed six lanes; bottom gave each lane its own
4-basis+null and contracted only their final sum. Added schema37 ABI, gradient
diagnostics, logger/checkpoint updates and three-direction review.

【观察到的结果】

Review caught BF16 public reconstruction error, 14% typed-common leakage after K
normalization, a dead W1 near-field decode graph, unnecessary Teacher diagnostics,
and old-schema audit semantics; these were repaired before release. Final local
checks reported 203 tests plus Ruff, compileall and diff check. Static lifecycle
kept Teacher train-only, S/W/static P1 once per observation and dynamic P2/P3/
bottom at five steps plus endpoint.

【当时的解释】

The release aimed to recover information without forcing amplitude or adding
capacity/loss. Passing static review was explicitly not a claim of experimental
benefit.

【后来怎么看（截至冻结终点）】

Partially retained. It fixed many axis losses, but its live run revealed that
dynamic precision bandwidth and W terminal adoption were still not closed.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；信息接口改善真实，动作消费闭环未完成。

【证据与锚点】

- 类型：日志观测；测试/运行观测；推断/未决
- 对话：visible messages 5105-5145。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Schema37, information axes, six lanes and late terminal aggregation.

### CI-0394 — Schema37 fixed object/type/camera structure but still lost dynamic precision and most W interval variation

【时间 / 版本】

Schema37 batch 1440-2400, 2026-08-24.

【当时在解决什么问题】

Verify whether restored information axes were actually used by the action path.

【当时的 insight / hypothesis】

Axis preservation and nonzero gradients are not enough: the supervised terminal
must preserve producer variation and each legal source needs sufficient action
bandwidth.

【具体做法】

Confirmed run identity and aligned source, Teacher, W latent, W field, P2,
consequence, P3, bottom and action with V120 and V36 where semantics matched.

【观察到的结果】

Schema37 improved S typed context 44%, appearance context 8.9x, temporal
variation 4.7x and W object cosine roughly `.41→.11`; K×C geometry and six
independent lanes were active. Yet P3 precision was only `.017-.0355` versus
V120 `.232-.259`, because dynamic P1 `.142` was query-only while static fact
was `.0143`. W interval variation `.0125-.0149` was only 27-36% of Teacher
`.0416-.0462`, adjacent cosine `.969-.980`, and common/residual ratio 5.2x
versus Teacher 2.8x. Public/typed/W P2 scores were about `.676/.130/.068`.
The action curve was close to V36 but 13-17% behind V120 in arm/long horizon.

【当时的解释】

Schema37 solved object-axis and routing-competition problems but inherited the
Schema36 overcorrection that removed action-conditioned precision. W's temporal
producer/consumer closure also remained weak; comparing it only to V36 had
initially understated the absolute failure.

【后来怎么看（截至冻结终点）】

Retained. This was the evidence that the next version had to close action
consumption rather than merely preserve interfaces.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5150-5165。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.676/.130/.068`。

【关联】

Schema37 live evidence, dynamic precision loss and W temporal publicization.

### CI-0395 — W's repeated publicization survived because reviews stopped at its latent, not its supervised field and action consumer

【时间 / 版本】

Schema37 W terminal diagnosis, 2026-08-24.

【当时在解决什么问题】

Explain why W publicization had been "fixed" repeatedly but still appeared
severely in Schema37.

【当时的 insight / hypothesis】

The functional invariant is that the exported four-interval FutureEffect retains
meaningful Teacher variation and changes action; internal interval differences
alone do not satisfy W's contract.

【具体做法】

Compared W typed latent and W2 internal per-type variation with decoded
FutureObjectDynamics and P2 scores.

【观察到的结果】

Typed interval hidden RMS was about `.61` and W2 internal variation about
`.19-.45`, while exported interval variation was only `.0125`. Past versions
had repaired S inputs, four intervals, far gauge, near/far direction, status,
object/type/camera axes and latent supervision. They had not required the shared
zero-initialized `delta_head` and final field readout to preserve that difference,
nor required P2 to depend on it. Common and interval states shared the same
readout and the easy common solution remained available.

【当时的解释】

The loop persisted because each review proved "W contains differences" or
"P2 can read W," not the composed invariant "differences survive W output and
are consumed by action."

【后来怎么看（截至冻结终点）】

Retained as the clearest example of a missing terminal functional invariant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 5155-5165。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`delta_head`。

【关联】

W decoder bottleneck, terminal adoption and repeated false closure.

### CI-0396 — The early Schema37 gradient spike was recurrent but not yet source-attributable

【时间 / 版本】

Schema37 epoch-1 batch 2220-2360, 2026-08-24.

【当时在解决什么问题】

Interpret a logged `global_preclip=54.96` window without repeating the mistake
of assigning a window statistic to the last sampled parameter owner.

【当时的 insight / hypothesis】

Window averages and current-batch parameter samples cannot locate a spike.
Instrumentation must capture the triggering batch and parameter at event time.

【具体做法】

Examined neighboring 20-batch windows and separated observed recurrence from a
causal source claim.

【观察到的结果】

Normal median was about `1.22`, while windows at b2220/b2280/b2360 averaged
`6.63/54.96/3.25`. This implied at least one extreme finite spike or several
moderate spikes in the b2280 window. It was intermittent and repeated, not a
stable high-gradient phase or proven periodic event. The existing owner sample
could not identify its parameter.

【当时的解释】

The spike was a P0 observation requiring per-batch event logging, but it was not
yet evidence that Schema37 introduced it or that W/G caused it.

【后来怎么看（截至冻结终点）】

Retained with uncertainty. Similar historical explosions cannot be assumed to
share this cause.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：部分闭环
- 说明：现象确定；来源未定。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 5163-5171。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`6.63/54.96/3.25`。

【关联】

Gradient event instrumentation and window-versus-batch semantics.

### CI-0397 — Similar "upstream information, weak W marginal use" patterns recurred historically, but a single shared cause was not proven

【时间 / 版本】

V98/V113/V117-118 ancestry replay, 2026-08-24.

【当时在解决什么问题】

Check the user's memory that the current G/Teacher→W→P failure chain had appeared
before.

【当时的 insight / hypothesis】

Historical interventions can establish recurring functional patterns without
proving identical mechanisms across architectures.

【具体做法】

Revisited flow-address, G/DINO, world/W, temporal and P2 intervention results and
then explicitly audited the certainty of the cross-version inference.

【观察到的结果】

V98 flow changed reader representation but action delta was only `.000699`.
V113 showed G3-address zero and DINO shuffle harming action about 10%/12.7%,
while zeroing world residual improved about 10.7% and W1 delta zero improved
2.3%; temporal and P2 episode shuffles caused large harm. V117/118 showed public
W slots, losses bypassing the route, diffuse Teacher and public/fixed temporal
selection. These resembled "upstream evidence exists, W marginal contribution
is weak, downstream uses time/current-fact paths."

【当时的解释】

The first narration was too absolute. The interventions prove useful address,
a harmful W checkpoint, and important time signals separately; they do not prove
one invariant causal mechanism. W also should not monopolize all action evidence.

【后来怎么看（截至冻结终点）】

Retained only in the qualified form: recurring pattern, current source-specific
causes, distinct evidence strengths.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：过强统一因果已撤回；限定模式保留。

【证据与锚点】

- 类型：因果干预；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 5170-5178。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Historical interventions, recurring patterns and causal humility.

### CI-0398 — W and P2 became external effect docks through V115-V119 rather than one sudden refactor

【时间 / 版本】

V115-V120 architecture ancestry, 2026-08-24.

【当时在解决什么问题】

Locate when W stopped being a continuous world carrier and P2 stopped being an
ordinary policy block.

【当时的 insight / hypothesis】

The current optional-dock behavior was the cumulative result of a staged
architectural transition, not merely a Schema-era regression.

【具体做法】

Read historical scripts and V120 snapshot behavior rather than inferring from
module names.

【观察到的结果】

V115 replaced harmful monolithic `world_residual` with a supervised
`FutureEffectField`. V116 structured the spatial effect; V117 introduced a P2
effect reader that wrote into the trajectory before the ordinary P2 block. V119
froze/skipped the old penultimate policy block under explicit object top and
replaced it with effect reader plus zero-preserving consequence. Thus W began
changing role at V115 and became a P2-read effect bank at V117; P2 began
externalizing at V117 and became a dock at V119.

【当时的解释】

The transition addressed genuinely harmful world residual, but it also made W's
future effect and P2's organization structurally external to the continuous
policy state. Later null/public routes amplified that optionality.

【后来怎么看（截至冻结终点）】

Retained as ancestry, not an instruction to restore the old monolithic carrier.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为结构转折记录。

【证据与锚点】

- 类型：历史检索
- 对话：visible messages 5179-5186。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`world_residual`。

【关联】

W/P2 architecture ancestry and future-consequence dock semantics.

### CI-0399 — The next closure target and anchor roles were clarified; no historical version was a health ideal

【时间 / 版本】

Post-Schema37 preparation, 2026-08-24.

【当时在解决什么问题】

Summarize what the next important version must accomplish and choose appropriate
regression, performance and health anchors.

【当时的 insight / hypothesis】

Longstanding defects should be structurally reduced, introduced information
deletions must be corrected, and small deterministic bugs can join the same
closed module family. No old version should be copied as a health target.

【具体做法】

Defined the main task as making preserved G/S/W/P1 information necessarily and
legitimately consumed by the action path without hard gates, quotas, gains or
new losses. Included restoration of bounded dynamic precision, comparable
public/typed/W P2 roles, geometry bandwidth, P3 lane ownership, G3 gauge order
and event-time gradient attribution. Rechecked V120/V25/V26 roles against logs.

【观察到的结果】

V120 was confirmed as the regression/behavior-semantics anchor: it protects
legal modules, action bandwidth, lifecycle and bottom behavior, but has weak
language/S/W and late tail growth. V25 was the primary full-run performance
anchor (best/final physical RMSE `.0772/.0789`); V26 was an auxiliary corrected-
behavior performance anchor. No historical health anchor existed: every one had
meaningful structural debt. The new version was expected to set a new health
upper bound while preserving V120 integrity and V25/V26 performance potential.
An elaborate renaming proposal was rejected as premature and no code names were
changed.

【当时的解释】

"Like V120" and "better architecture" were both insufficient objectives. The
three independent axes were regression integrity, performance potential and
structural health.

【后来怎么看（截至冻结终点）】

Retained as the preparation boundary for the following closure implementation.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；命名方案已否定。

【证据与锚点】

- 类型：日志观测；方案/设计记录；推断/未决
- 对话：visible messages 5187-5214。
- 段级工件：见 Segment 039 boundary 的 `Primary historical artifacts`。

【关联】

Action-path closure, G/W/P roles and historical anchor taxonomy.

## Segment 039 synthesis

The next-day replay corrected the initial interpretation of Schema35's 55x P1
ratio. The raw dynamic dominance likely existed in V120; Schema35's regression
was to remove that carrier from protected consequence/transition/bottom while
giving it direct optional precision authority, and to place an internally
unbounded modulation/Jacobian behind a bounded forward output. The epoch-3
failure propagated through shared canvas and global clipping before the first
enumerated flow parameter became non-finite. This led to a genuine whole-graph
audit across online, supervision and optimization graphs. Schema36 removed the
duplicate precision authority and false status consumer, fixing the P0
instability, but swung too far: dynamic precision had only a query outlet.
Meanwhile W's severe early publicization was traced not to smaller W or missing
loss but to harder constrained residual learning and a downstream action path
that did not require it; later Schema36 evidence showed partial recovery and
forced a stage-aware qualification.

The information-conservation audit then separated necessary semantics from six
explicit losses across S→P1, W, camera geometry, P2 types, dynamic precision and
bottom routing. It formalized the key correction that ownership is not
exclusivity and a structural bottleneck must not delete information. Schema37
restored many axes: typed/goal/history S context, W far freedom, camera-aware P2,
typed consequence, six P3 lanes and independent bottom nulls. Its early log
proved those repairs active and improved W object identity, but also showed P3
precision one order below V120, W retaining only roughly one third of Teacher's
interval variation, and public S still greatly outscoring W at P2. W internal
latents contained substantial interval difference that the shared terminal
readout and action consumer did not preserve—explaining why years of local W
repairs had not closed the functional contract. A recurrent finite-gradient
spike also exposed the need for event-time, per-parameter attribution.

Historical replay then qualified, rather than flattened, ancestry: V98/V113/
V117-118 exhibited related weak-marginal-use patterns but not a proven single
mechanism; W transitioned from world carrier to supervised effect bank at
V115-117 and P2 became an external consequence dock by V119. The next version's
task was therefore action-consumption closure, not restoration of monolithic
world residual. V120 was fixed as the regression/behavior anchor, V25 as the
primary performance anchor, V26 as an auxiliary performance anchor, and no old
version was accepted as a structural-health ideal.

## Segment 040 — Schema38-39, terminal-axis closure, lawful priors and soft-address numerical ownership

### Segment boundary

```text
Date:                 2026-08-25
Visible messages:     5215-5362
Starts with:          full Schema37 evidence and preparation of action-consumption closure
Ends with:            Schema39 implementation, independent reviews and startup-registry repair
Primary historical artifacts:
  near-complete Schema37 log
  complete eight-epoch Schema36 log
  Schema38 epoch-1 log and validation paste
  SCHEMA39_DESIGN_CHECKPOINT.md
  V120 exact source and V25/V26 performance logs
  00_CURRENT_ARCHITECTURE_CONTRACT.md
  CURRENT_MAINLINE_ISSUES.md
```

### CI-0400 — Mature Schema37 narrowed the next repair to precision consumption, W temporal use and two deterministic algebraic errors

【时间 / 版本】

Schema37 through epoch 8 batch 1660, 2026-08-25.

【当时在解决什么问题】

Use the latest long run to correct the final implementation scope before touching
source again.

【当时的 insight / hypothesis】

When broad training is stable, the next version should target the terminal
consumer boundaries proven weak rather than reopen Teacher, flow, S capacity or
bottom.

【具体做法】

Compared mature V37 against V120 and V25/V26 anchors, then separated behavior
changes from gauge-equivalent and target-reachability repairs.

【观察到的结果】

V37 epoch-7 physical RMSE was `.08165`, with healthy `1.92 s/batch` and
`11.9 GiB`, but P3 precision was only `.0258` versus historical `.24-.29`.
W retained about 56% of Teacher residual and adjacent cosine was `.9765`.
Observation raw gradient spikes reached `44.67` but the window logger could not
locate their parameter. Geometry had no coordinate-unit mismatch; the concrete
problem was an unreachable positive covariance floor and the failure of geometry
to inform high-bandwidth semantic address. G3 correction required a softmax-
gauge-equivalent cleanup, not behavior-changing "center before tanh."

【当时的解释】

The implementation scope was fixed to complete W/P2 consumption, bounded dynamic
P1 precision, geometry/address closure, covariance reachability and exact spike
attribution. Teacher, flow, S producer, static P1 and bottom remained out of scope.

【后来怎么看（截至冻结终点）】

Retained as the corrected Schema38 boundary.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 5215-5223。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`1.92 s/batch`。

【关联】

Schema37 mature evidence and Schema38 scope.

### CI-0401 — Schema38 reconnected complete W fields and dynamic precision without adding a second carrier budget

【时间 / 版本】

Schema38 implementation, 2026-08-25.

【当时在解决什么问题】

Make action consumers read the full W interval field and restore legal dynamic
P1 precision while preserving static fact ownership.

【当时的 insight / hypothesis】

The existing `semantic_delta/transport_mean` already formed complete four-
interval fields; a new carrier was unnecessary. Dynamic policy could modulate
precision without becoming fact or receiving a second update budget.

【具体做法】

P2 selected `common+residual` complete fields, removed S's independent interval
logit, used S only to condition W keys, gave types local nulls, restored dynamic
P1 to fact-conditioned P3 precision, let geometry condition semantic K address,
removed the covariance lower-bound mismatch and cleaned G3 gauge equivalently.
Added per-batch spike attribution, release gates, migration validation and true
epoch-tail windows.

【观察到的结果】

Independent review caught duplicate K support in geometry correction, hot-path
counterfactual diagnostics, incomplete bottom-migration ABI checks and stale
metric aliases. These were fixed before release. Action-only gradients reached
W/S/P2/P3/dynamic P1/covariance after the zero-initialized W value was opened by
future supervision. Final static checks reported 244 passes. Parameters fell by
786,432 only because three independent S voting matrices were removed; bottom
remained 55,832,880 parameters.

【当时的解释】

This was a consumer-algebra repair, not a new mechanism. The two-stage gradient
boundary at zero-initialized W was explicitly documented rather than mislabeled
as a dead path.

【后来怎么看（截至冻结终点）】

Partially retained. The complete-field and dynamic-precision repairs worked, but
deleting S's independent temporal prior and terminating interval at P2 created a
new half-closure.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留。

【证据与锚点】

- 类型：源码事实；日志观测；因果干预；推断/未决
- 对话：visible messages 5224-5249。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`semantic_delta/transport_mean`。

【关联】

Schema38, complete W field, dynamic precision and release audits.

### CI-0402 — Complete V36 proved behavior can recover while the future/precision structure remains unhealthy

【时间 / 版本】

Schema36 complete eight-epoch audit, 2026-08-25.

【当时在解决什么问题】

Determine whether V36 was a failed run or a behaviorally competitive but
structurally compensated baseline.

【当时的 insight / hypothesis】

Aggregate RMSE can be restored by healthy bottom/current-fact paths while W
future ownership and precision consumption remain underused.

【具体做法】

Compared all eight epochs with V120 and V25 under the same action normalizer and
separated arm, gripper, normalized, W/P2, P1, flow, execution and gradients.

【观察到的结果】

V36 eight-epoch mean/best/final physical RMSE was
`.08523/.08007/.08106`, versus V120 `.08628/.07931/.08145` and V25
`.08689/.07725/.07887`. V36 final arm `.05983` was strong but gripper `.15659`
lagged. W object cosine near `.10` was healthy, yet interval variation `.0336`
was only 55% of Teacher and S public/typed/W scores were `.592/.485/.106`.
Dynamic/static P1 was 18x while precision remained below V25/26. Training flow
continued down while validation flow worsened late; RMSE rebounded 1.25% after
epoch 6. Sparse finite observation spikes occurred, but no explosion.

【当时的解释】

V36 was a valuable behavior baseline, not a health target. It demonstrated that
arm/current-state compensation could mask weak W time use and misplaced dynamic
precision.

【后来怎么看（截至冻结终点）】

Retained. It justified preserving the mainline while continuing structural
closure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为行为竞争、结构不健康的对照。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5250-5256。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.08523/.08007/.08106`、`.08628/.07931/.08145`、`.08689/.07725/.07887`、`.592/.485/.106`。

【关联】

Behavior versus structural health and V36 full-run evidence.

### CI-0403 — Parameter-level spike logging separated harmless decoder cold start from later observation/address instability

【时间 / 版本】

Schema38 early spike audit, 2026-08-25.

【当时在解决什么问题】

Interpret new gradient-spike lines and decide whether all spikes belonged to one
failure chain.

【当时的 insight / hypothesis】

Finite preclip spikes must be classified by event time, owner and local-clip
boundary; a common threshold does not imply a common cause.

【具体做法】

Examined each event's global norm, maximum-L2 parameter, maximum element and
postlocal/global clipping semantics.

【观察到的结果】

The first events at batches 6/9/11 had global norms `5.02/5.89/5.67`, with
93-95% of squared gradient in `bottom.decoder.velocity_head.arm_abs.weight` and
element maxima `.43-.50`. Decoder-local clipping reduced these before global
clip, consistent with cold-start output calibration. Later Schema38 logs instead
contained rare observation/flow-address events: about 0.85% of batches, median
12.78, eight above 20 and three above 100, with a maximum 1763.72 at batch 808,
`target_dino_key` L2 1134 and flow-delta max-abs 92.

【当时的解释】

The early arm-head events were expected and contained. The later address events
were a separate numerical chain requiring low-support/variance analysis; they
were not evidence of W publicization.

【后来怎么看（截至冻结终点）】

Retained. Precise event logging replaced misleading window-owner attribution.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 5257-5259 and 5265-5292。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`bottom.decoder.velocity_head.arm_abs.weight`、`target_dino_key`。
- 直接量值/表达式（卡内原文）：`5.02/5.89/5.67`。

【关联】

Gradient spike taxonomy, decoder local clip and soft-address stability.

### CI-0404 — Schema38 closed old consumers only halfway: uniform interval selection cancelled almost all zero-mean residual

【时间 / 版本】

Schema38 batch 1100 through epoch-1 validation, 2026-08-25.

【当时在解决什么问题】

Verify whether complete-field selection and removal of S's independent vote made
W interval differences reach action.

【当时的 insight / hypothesis】

`uniform posterior × zero-mean residual = zero` is a deterministic degeneracy.
Consumer closure must be tested under the cheapest legal posterior, not assumed
to emerge from training.

【具体做法】

Aligned Schema38 with historical runs, inspected interval masses/null/retention,
and followed W residual through the exact terminal formula.

【观察到的结果】

Action training was normal and epoch-1 physical RMSE `.09854` was within 1% of
V120 and 4.2% better than V37. S independent vote was zero and dynamic precision
rose to `.062`, but W interval masses were nearly uniform
`.1427/.1426/.1388/.1390`, type-local null reached `.437`, and 96.7-97.0% of
residual was cancelled; selected residual/common was only 1.6-1.7%. W adjacent
cosine remained around `.98-.993`. P3 precision was still far below V25/26.

【当时的解释】

Schema38 proved that the old path was blocked, but not that the replacement
could carry temporal innovation. It left action mainly on W common plus existing
P1/bottom paths.

【后来怎么看（截至冻结终点）】

Retained as the decisive argument for preserving interval until P3 rather than
terminating it in P2.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；Schema38 半闭环。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 5260-5269 and 5312-5315。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`.1427/.1426/.1388/.1390`。

【关联】

Uniform-posterior degeneracy, zero-mean residual and terminal-axis placement.

### CI-0405 — An independent S temporal prior was lawful; deleting it was an unsupported overcorrection

【时间 / 版本】

Schema38 causal-language correction, 2026-08-25.

【当时在解决什么问题】

Challenge the claim that S's independent interval influence was an "illegal
bypass."

【当时的 insight / hypothesis】

Independence, dominance and illegality are different claims. Goal/history may
legitimately provide a temporal prior even though only W supplies grounded
future effect value.

【具体做法】

Separated what interventions and logs proved from the normative architecture
choice, then restated the desired probabilistic roles.

【观察到的结果】

It was proven that S could alter P2 interval posterior with W fixed and that
historically S scores dominated W. That did not prove S should have no temporal
vote. The correct relation was: S provides a bounded prior, W provides interval-
conditioned effect likelihood/value, and P2 combines them. W zero must still
yield zero effect, but S may express time preference. Schema38's complete removal
of the prior therefore deleted a legal function.

【当时的解释】

The word "bypass" had been used as a design verdict without proof. This was
another case of treating imbalance as evidence that a function should not exist.

【后来怎么看（截至冻结终点）】

Retained. The earlier count of "at least seven" similar overcorrections was an
informal lower-bound taxonomy, not a rigorous version count.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：过强定性已撤回；prior/likelihood/value 分工保留。

【证据与锚点】

- 类型：因果干预；推断/未决
- 对话：visible messages 5268-5283。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

Lawful priors, causal language and overcorrection discipline.

### CI-0406 — Source-first re-audit showed bottom lanes were already independent and isolated three distinct remaining chains

【时间 / 版本】

Schema38 source-first audit, 2026-08-25.

【当时在解决什么问题】

Replace an untrusted architecture contract with the actual forward calls and
determine what was still wrong without declaring the whole version failed.

【当时的 insight / hypothesis】

The contract should be treated as a falsifiable claim when it has repeatedly
encoded desired semantics rather than execution facts.

【具体做法】

Rebuilt P2→P3→bottom and the action-only reverse path directly from source, then
separated stable information loss from rare numerical spikes.

【观察到的结果】

Bottom already gave each of six P3 lanes its own 4-basis+null before summation;
lane competition was not the current fault. Three independent chains remained:
(1) rare flow/address gradient spikes; (2) stable W residual cancellation when
P2 terminated interval; and (3) dynamic precision ownership/scale. At batch
2360, action-flow median and native mean were essentially V120-like, G slots and
flow were learning, and bottom lanes were active. The version was not globally
broken.

【当时的解释】

The audit corrected both the contract and the emotional overreaction: a serious
local closure fault did not imply total model failure, and the spike did not
share W's stable cancellation mechanism.

【后来怎么看（截至冻结终点）】

Retained. Emotional events were used to tighten evidence language, not to judge
the participant.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5276-5292。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

Source-as-fact review, independent failure chains and calibrated severity.

### CI-0407 — Schema38 exposed additional amplifiers and false-positive tests, but they did not all warrant redesign

【时间 / 版本】

Schema38 extended audit, 2026-08-25.

【当时在解决什么问题】

Find secondary causes around the two terminal failures without expanding into a
new G/S/W/P rebuild.

【当时的 insight / hypothesis】

An amplifier should be fixed only when its source algebra is determined; a high
null or large dynamic query remains a hypothesis until an intervention proves
its action effect.

【具体做法】

Checked S key conditioning, W common formation/readout, null scope, dynamic P1
query addition, complete-field diagnostics and action-gradient tests.

【观察到的结果】

`normalize(W_key * (1+tanh(S)))` erased mostly radial S modulation; S intervention
changed posterior only `.00336`. W interval hidden mean fed common and shared
the decode head, favoring common. Type null could reject both common and residual
(semantic 54%, geometry 40%), but shortcut status remained unproven. Dynamic/
static query inputs differed around 14x. BF16 identity diagnostics were false
errors, a residual-retention test allowed zero retention, and the gradient test
only required one parameter in a module to receive gradient.

【当时的解释】

The definite fixes were the normalized-away modulation and false tests. Null
shortcut and dynamic posterior dominance required more evidence; they could not
justify deleting another carrier.

【后来怎么看（截至冻结终点）】

Retained as the evidence boundary used to narrow Schema39. Confirmed amplifiers
were handled selectively; unproven items remained evidence gaps rather than
authority for an undifferentiated redesign.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；confirmed items 被 Schema39 吸收，其余 hypothesis 在冻结点仍未验证。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；因果干预；推断/未决
- 对话：visible messages 5293-5305。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

S key modulation, null scope, W readout and adversarial closure tests.

### CI-0408 — Schema39 moved temporal termination from P2 to P3 and created one protected policy-precision carrier

【时间 / 版本】

Schema39 reviewed design, 2026-08-25.

【当时在解决什么问题】

End the cycle without reimplementing axes and modules already restored in
Schema37/38.

【当时的 insight / hypothesis】

P2 may legally terminate K/camera spatial selection, but terminating interval
before action compilation destroys signed temporal information. Dynamic P1
needs a protected policy carrier distinct from fact and sharing the single total
update budget.

【具体做法】

The main agent formed one proposal; subagents were restricted to adversarial
review. The reviewed design made P2 output explicit `[B,24,4,4,type,H]` selected
interval evidence, let P3 perform the no-null physical temporal terminal before
consequence, and routed dynamic P1 through a shared-parameter 4-basis/no-null
protected carrier. It also treated the soft-address numerical chain—producer
FP32 log measures, low support and variance scales—as one boundary.

【观察到的结果】

Reviews corrected three details: preserve explicit 24×4×4×type axes, select time
before consequence, and avoid a second dynamic-precision budget. They also found
that changing isolated square roots was insufficient; grounder log measure, P2
low-support logs and G2 correction had to agree.

【当时的解释】

The plan changed two terminal interfaces and one numerical ownership boundary,
not the whole top. It retained Teacher/W targets, S/W producers, static P1 and
bottom body.

【后来怎么看（截至冻结终点）】

Retained as the implemented Schema39 design.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已实施。

【证据与锚点】

- 类型：过程/决策记录
- 对话：visible messages 5295-5315。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

Terminal-axis ownership, protected policy precision and adversarial design review.

### CI-0409 — Schema39 carried FP32 probability ownership end to end instead of reconstructing it after BF16 loss

【时间 / 版本】

Schema39 implementation, 2026-08-25.

【当时在解决什么问题】

Prevent low-support soft-address/camera evidence from producing fake priors or
unbounded gradients after BF16 rounding.

【当时的 insight / hypothesis】

`clamp→log` at a consumer cannot recover a probability that a producer already
rounded or underflowed. The producer must own and export the FP32 log measure.

【具体做法】

Propagated FP32 owner log posterior from G2/G3 through LocalFactSet, Grounder,
K+null, typed/camera aggregation, W camera availability/weights and P2. Replaced
near-zero variance square-root sensitivity with a zero-preserving variance scale
while retaining epsilon baseline. Added tiny-camera/all-invalid counterexamples,
finite/PSD typed invariants and action-only gradient tests.

【观察到的结果】

The review caught two half-fixes before release: G exported typed owner back to
BF16 before Grounder, and W later cast camera availability/weights back to BF16
before P2. Both were removed. Tiny legal mass no longer became a clamp-created
uniform prior. Dynamic P1 was verified to enter the protected carrier exactly
once and transition only through a bounded action operand.

【当时的解释】

This was numerical information conservation: preserve probability measure and
its semantics at the producer instead of patching each downstream symptom.

【后来怎么看（截至冻结终点）】

Retained. It was the source-level response to the observation/address spike chain.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已实施并保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测
- 对话：visible messages 5316-5336。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

FP32 log measure, low-support stability and producer ownership.

### CI-0410 — Independent review had to include runtime tools and public semantics, not only model source

【时间 / 版本】

Schema39 release review, 2026-08-25.

【当时在解决什么问题】

Ensure a source-correct model would not be mislabeled, wrongly gated or silently
migrated by stale operational tooling.

【当时的 insight / hypothesis】

Experiment validity is part of the architecture boundary: logger, recovery
audit, checkpoint identity, migration and public documentation must describe the
same graph as the model.

【具体做法】

After the main implementation froze at 209 tests, independent agents reviewed
axes/bypasses, gradients/numerics and runtime/checkpoint/logs without editing.
The main agent fixed only reported concrete gaps and reran suites.

【观察到的结果】

Review added FP32/finite/PSD validation to typed interfaces; corrected README and
package exports; updated recovery audit from Schema38 shared/status/null metrics
to Schema39 spatial-P2/temporal-P3 carriers; prevented bottom-owned spikes from
printing irrelevant flow attribution; removed hard-coded Schema23/24 gate debt;
and removed duplicate transition/precision diagnostics. Final relevant suites
reached 231 passes, with no remaining P0/P1 runtime or checkpoint finding.

【当时的解释】

A stale recovery tool was a release blocker even though it did not change
forward values, because it could accept the wrong run or reject the right one.

【后来怎么看（截至冻结终点）】

Retained as the required operational half of closure.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；测试/运行观测；过程/决策记录；推断/未决
- 对话：visible messages 5336-5355。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。

【关联】

Runtime semantics, checkpoint identity, logging and independent review roles.

### CI-0411 — Schema39's launch failed because a renamed module key had two registries; the fix made one registry authoritative

【时间 / 版本】

Schema39 first server launch, 2026-08-25.

【当时在解决什么问题】

Repair a preflight `KeyError: 'future_effect_p2'` after the model key had changed.

【当时的 insight / hypothesis】

A producer/consumer rename is not closed if display summaries maintain their own
hard-coded registry. Startup execution must be tested with the production model,
not only a reduced fixture.

【具体做法】

Traced module parameter registry→serialization→console summary, replaced the
separate old-key list with a single `(key, label, module)` registry, and ran the
exact production summary line with default model construction.

【观察到的结果】

The model had registered `future_effect_p2_spatial_p3_terminal`, while startup
still indexed `future_effect_p2`. Training had not begun; the error occurred
before run context and did not involve numerical/data/memory behavior. After the
fix, production summary printed `P2spatial/P3terminal=2132995`, total parameters
`168687044`, and 156 related tests plus static checks passed.

【当时的解释】

This was a deterministic half-renaming error that the declared release review
should have caught. A single authoritative registry removed the error class
rather than patching one print line.

【后来怎么看（截至冻结终点）】

Retained as a release-process correction.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：已修复。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 5356-5362。
- 段级工件：见 Segment 040 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`future_effect_p2_spatial_p3_terminal`、`future_effect_p2`。
- 直接量值/表达式（卡内原文）：`P2spatial/P3terminal=2132995`、`168687044`。

【关联】

Startup ABI, single registry and production-path testing.

## Segment 040 synthesis

Mature Schema37 evidence narrowed the next change to action consumption,
dynamic precision, geometry/address reachability and exact gradient attribution.
Schema38 correctly reconstructed complete W fields, restored dynamic P1 to a
fact-conditioned precision lane and added parameter-level spike logging. It also
proved that complete V36 had recovered V120-like behavior while retaining
structural debt: W time use, dynamic precision placement and gripper/generalization
remained weak. Precise spike logs then separated benign early arm-head calibration
from rare, very large observation/address events.

Schema38's first live run exposed another half-closure. With nearly uniform
interval weights, a zero-mean W residual is algebraically cancelled; about 97%
of temporal residual disappeared despite normal action loss. The user correctly
challenged the unsupported label "illegal S bypass": S goal/history may supply a
lawful bounded temporal prior, while W alone supplies grounded future value.
Deleting the prior was another overcorrection. A source-first audit further
showed bottom lanes were already independent and isolated three distinct chains:
temporal residual termination, dynamic precision ownership and rare soft-address
spikes. Secondary review found S modulation normalized away, broad type-local
null scope and tests that passed with zero residual retention.

Schema39 therefore moved only the real terminals: P2 ended K/camera space while
preserving four intervals; P3 performed the no-null physical temporal terminal;
dynamic P1 received one protected policy-precision carrier without becoming
fact or gaining a second budget. The numerical repair preserved producer-owned
FP32 log measures across G, Grounder, W camera support and P2, rather than trying
to reconstruct tiny probabilities after BF16. Independent reviews covered model,
runtime, checkpoint, logger, recovery audit and public semantics. One launch-
blocking key rename still escaped and was fixed by consolidating all module
summary consumers into a single production-tested registry—an important reminder
that release closure includes startup code, not only forward/backward tests.

## Segment 041 — Schema39 live evidence, anchor correction and the first structural break timeline

### Segment boundary

```text
Date:                 2026-08-26
Visible messages:     5363-5395
Starts with:          detailed V39/V38 live-log comparison
Ends with:            why V27's three ownership repairs multiplied into information deletion
Primary historical artifacts:
  Schema38 and Schema39 full/partial training logs
  V25/V26 performance logs
  v120_long.log
  history_log/ V27-V29 logs
  docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md
```

### CI-0412 — Parent-version comparison identifies regressions, but cannot replace performance and health anchors

【时间 / 版本】

Schema39 live evaluation, 2026-08-26.

【当时在解决什么问题】

Judge whether Schema39's new action-consumption closure actually improved the
model.

【当时的 insight / hypothesis】

A comparison has different meanings at three levels: V38 identifies changes
introduced by its child, V25/V26 anchor achieved performance, and V120 anchors
older main-path health. Treating the immediate parent as the only baseline can
turn a local improvement into a false overall-success claim.

【具体做法】

The first audit aligned V38 and V39 by iteration and epoch. After the user
challenged that scope, the audit was redone against V25/V26 and V120, while V38
was retained only for attribution of new regressions or recoveries.

【观察到的结果】

The corrected comparison changed the interpretation. At V39 epoch 7, normalized
trajectory error was near the V25/V26 level and arm/first/tail were not the main
gap, but physical RMSE remained about 8-11% worse and gripper physical error
about 18-24% worse. Relative to V120, physical RMSE was about 5% worse than its
final point and 8% worse than its best point.

【当时的解释】

The apparent physical/normalized contradiction came from physical gripper
calibration and event timing carrying a larger de-normalized cost, not from a
uniform degradation of the entire arm trajectory.

【后来怎么看（截至冻结终点）】

Retained, including the explicit correction of the initial V38-centric reading.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测；过程/决策记录；推断/未决
- 对话：visible messages 5364-5374。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。

【关联】

Baseline roles, experiment attribution and gripper-specific validation.

### CI-0413 — Schema39 closed the consumer topology but retained a severe temporal-bandwidth and gripper gap

【时间 / 版本】

Schema39 epochs 1-7, 2026-08-26.

【当时在解决什么问题】

Separate a bad W/P consumer implementation from a weak upstream future signal.

【当时的 insight / hypothesis】

Object identity and temporal identity are independent health axes. A lower
object-pair cosine can coexist with four nearly identical interval fields, so
"W is more public" is too coarse a diagnosis.

【具体做法】

Verified the Schema39 source topology—P2 no-null K/camera spatial selection,
P3 as the sole four-interval terminal, common/residual each read once and zero
identity error—then compared object-pair cosine, adjacent-interval cosine,
selected residual/common and decoded action metrics with the three anchors.

【观察到的结果】

- Schema39 removed the V38 complete-field null rejection and increased selected
  interval residual relative to common.
- W object-pair cosine was about `0.282`, healthier than V120 (`~0.436`) and
  V25/V26 (`~0.473/~0.555`).
- W adjacent-interval cosine was about `0.960`, worse than V120 (`~0.921`) and
  V25/V26 (`~0.909`).
- At epoch 7, arm physical and long-horizon normalized bands were competitive,
  while gripper physical was `0.16929` versus `0.13713/0.14330` in V25/V26;
  decoded gripper F1 was also lower.

【当时的解释】

Schema39 made W information enter the action main path more honestly, but did
not make the produced interval evidence sufficiently distinct. The remaining
failure was temporal consequence bandwidth and downstream gripper use, not an
all-axis W collapse or another hidden P2 null bypass.

【后来怎么看（截至冻结终点）】

Retained as a mixed result: consumer closure succeeded, upstream temporal
production and fine action calibration did not recover.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分成功；问题未闭环。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 5366-5376。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`~0.473/~0.555`。

【关联】

Teacher interval targets, W temporal identity, P3 terminal selection and gripper.

### CI-0414 — Rare late Schema39 spikes shifted attribution toward observation/address producers, but did not prove a root cause

【时间 / 版本】

Schema39 epoch 8 partial log, 2026-08-26.

【当时在解决什么问题】

Determine whether the run had a persistent gradient explosion or isolated
address-chain events.

【当时的 insight / hypothesis】

The parameter with the largest gradient identifies the first visible victim or
dominant contributor for that batch, not necessarily the causal origin of the
event.

【具体做法】

Compared spike timestamps, global preclip norm, stable neighboring windows and
the reported max-L2/max-abs parameter between V38 and the partial V39 epoch 8.

【观察到的结果】

V39 contained two preclip events around `16-18`. Their dominant parameters had
moved from the early decoder arm head to the observation DINO address key; one
event also reported a large flow-delta-head scalar. Neighboring 20-batch windows
returned quickly to roughly `0.4-0.5`.

【当时的解释】

This ruled out a sustained global explosion but preserved a narrower numerical
or data-dependent observation/address investigation. Neither the named address
key nor the flow head was declared the root cause from this evidence alone.

【后来怎么看（截至冻结终点）】

Retained as localized evidence, not a causal verdict.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：部分闭环
- 说明：未完全定位。

【证据与锚点】

- 类型：日志观测；推断/未决
- 对话：visible messages 5369-5370。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。

【关联】

Producer-owned FP32 measures, non-finite/spike attribution and batch-level data probes.

### CI-0415 — The post-V25 history contains several distinct breakpoints, not one continuously worsening defect

【时间 / 版本】

V25-Schema39 retrospective, reconstructed 2026-08-26.

【当时在解决什么问题】

Locate when the reusable V25/V26 action path first broke and distinguish it from
later Teacher and P1 failures.

【当时的 insight / hypothesis】

"When did the problem start?" must be decomposed by mechanism. A consumer-chain
break, a Teacher-target collapse and a precision-ownership imbalance can begin
in different versions and later amplify one another.

【具体做法】

Parsed V27-V39 history logs under one metric vocabulary and traced the matching
source changes, separating first appearance from first severe manifestation.

【观察到的结果】

- V25/V26 remained usable despite W adjacent-interval cosine near `0.91`.
- V27 was the first clear consumer-chain break: P2 null mass rose from about
  `0.077` to `0.410`, W adjacent-interval cosine from `0.909` to `0.981`, W
  interval variation fell from `0.0839` to `0.0482`, and P3 precision/state
  change collapsed to about `1e-4`.
- V28/V29 amplified the P2-null and W-time collapse.
- V31 was a second, independent break: the background-centred partial-OT Teacher
  moved dustbin from about `0.047` to `0.516` and semantic delta from about
  `0.537` to `0.173`.
- V35 exposed a roughly 51-fold static/dynamic P1 scale imbalance.
- V37-V39 repeatedly repaired consumers while future-lane bandwidth remained weak.

【当时的解释】

The current failure was inherited from a sequence of distinct changes. Blaming
everything on V27, or treating V31's dustbin jump as the original consumer
failure, would prescribe the wrong repair.

【后来怎么看（截至冻结终点）】

Retained as a reconstructed timeline; individual causal labels remain bounded
to the source/log evidence cited at the time.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测
- 对话：visible messages 5389-5393。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。

【关联】

V27 ownership changes, V31 Teacher OT, V35 P1 ownership and Schema39 residual debt.

### CI-0416 — V27's "ownership purification" failed because three local deletions formed one serial information bottleneck

【时间 / 版本】

V27 retrospective, 2026-08-26.

【当时在解决什么问题】

Explain why three individually plausible changes were allowed in one version
and why their combined damage escaped review.

【当时的 insight / hypothesis】

Removing a shortcut is safe only if the legal information adjacent to it remains
expressible. Risks on a serial path multiply; they cannot be reviewed as three
independent local cleanups.

【具体做法】

Reconstructed the intent and realized algebra of the three changes:

- candidate-count calibration made the whole P2 W path approximately a 1:1
  real-versus-null option;
- removal of W interval identity removed the temporal reference frame under weak
  supervision;
- P3 basis centering removed common fine precision along with duplicate fact.

【观察到的结果】

The resulting chain was `weak W temporal identity → frequent P2 rejection →
common precision deleted at P3`. The V27 change covered 26 files and roughly
`+11684/-520` lines. Shape, normalization, zero semantics and nonzero-gradient
tests passed, while no test measured representational preservation or
`W→P2→P3→decoded gripper` retention.

【当时的解释】

The common failure mode was overcorrection: solve a local bias by deleting the
larger information channel around it. The review checked local legality but not
information conservation or the product of consecutive attenuations.

【后来怎么看（截至冻结终点）】

Retained as one of the clearest examples of why semantic cleanup cannot be paid
for with information loss.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：已否定原实现；工程教训保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 5393-5395。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`+11684/-520`。

【关联】

Information-conservation review, multiplicative bottlenecks and module-sized changes.

### CI-0417 — Loss of trust converted an informal diagnosis into an evidence-bounded handoff artifact

【时间 / 版本】

Schema39 handoff, 2026-08-26.

【当时在解决什么问题】

Make the work independently reviewable after repeated half-closures had made
verbal assurances insufficient.

【当时的 insight / hypothesis】

A handoff must let another task reproduce the evidence without inheriting the
author's confidence. Source-proved facts, log-proved observations and unresolved
hypotheses therefore need separate labels.

【具体做法】

Froze the model state and created `ACTIVE_MAINLINE_HANDOFF.md` with commit/schema
identity, log completeness, anchor roles, reproduction commands, the continuous
Teacher-to-gripper problem chain, unresolved assumptions and prohibited patch
patterns. The later V27/V31 breakpoint timeline was added to it.

【观察到的结果】

No model, contract, issue ledger or experiment script was changed. Schema39 was
explicitly marked incomplete at seven full epochs plus epoch 8 batch 1620. The
handoff did not promote the unproven cause of high Teacher dustbin to fact.

【当时的解释】

The user's loss of confidence materially raised the delivery standard from a
summary answer to a reproducible, independently auditable artifact.

【后来怎么看（截至冻结终点）】

Retained as a process correction; emotional context is recorded only because it
changed the evidence and handoff requirements.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；过程/决策记录
- 对话：visible messages 5382-5388 and 5392-5393。
- 段级工件：见 Segment 041 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`ACTIVE_MAINLINE_HANDOFF.md`。

【关联】

Handoff discipline, trust recovery and evidence classification.

## Segment 041 synthesis

Schema39's live run confirmed that its consumer topology was substantially more
honest than Schema38: P2 ended only spatial identity, P3 ended time, and the old
complete-field null rejection disappeared. The first comparison nevertheless
used the wrong primary anchor. Re-evaluation against V25/V26 for performance and
V120 for health showed a concentrated gripper/physical gap and severe temporal,
not object, publicization. This distinction prevented a broad but incorrect
claim that all W identities had collapsed.

The retrospective then separated a serial history into real breakpoints. V27
simultaneously weakened interval identity, made W globally rejectable and
deleted common precision; V31 independently collapsed Teacher association
bandwidth; V35 exposed precision-scale ownership problems. These were not one
defect with a single start date. The key engineering correction was to judge
whether legal information survives an ownership repair and to measure retention
across the whole serial consumer path. Repeated half-closures and a corresponding
loss of trust also led to a formal evidence-bounded handoff, making reproduction
and unresolved assumptions part of delivery rather than relying on narrative
confidence.

## Segment 042 — Pre-V27 ancestry and source-bounded architecture replay

### Segment boundary

```text
Date:                 2026-08-26
Visible messages:     5396-5428
Starts with:          tracing the break before V27
Ends with:            adding Schema37-39 individually to the architecture replay
Primary historical artifacts:
  local V120 snapshot and v120_long.log
  V121/V122 commits and logs where available
  independent-mainline creation commit and early/full log
  Schema20-Schema39 history
  ClearVLA_V120_to_V25_architecture_replay.md
  ClearVLA_architecture_replay_ledger.md
  docs/research/auxiliary/ARCHITECTURE_REPLAY_LEDGER.md
```

### CI-0418 — V120 had structural debt, while V121 introduced a mathematically inconsistent Teacher objective

【时间 / 版本】

V120-V121 retrospective, 2026-08-26.

【当时在解决什么问题】

Distinguish inherited V120 weaknesses from regressions introduced before the
independent mainline.

【当时的 insight / hypothesis】

An imperfect baseline and an internally inconsistent target are categorically
different. The former may still provide a valid behavioral path; the latter has
no exact optimum even with unlimited capacity.

【具体做法】

Compared the V120 snapshot/log with V121 commit `480f71c`, especially the two
Teacher targets and the online reconstruction relation expected by W.

【观察到的结果】

V120 retained non-collapsed G/W object axes, S interval/temporal variation around
`0.1445/0.0910`, P2 effect RMS about `0.0975`, P3 precision about `0.254` and best
physical RMSE `0.07931`, despite weak language reads, small G3 correction and W
temporal publicization. V121 defined successor content as a reliability-weighted
stable future while defining semantic delta from a different end-biased future,
yet W required `successor = current + delta`.

【当时的解释】

The V121 pair of targets could not simultaneously satisfy its own identity. This
was a source-proved semantic error, unlike V120's measurable but still usable
capacity/identifiability debts.

【后来怎么看（截至冻结终点）】

Retained. Absence of a complete V121 log prevents inventing a performance
magnitude, but does not weaken the algebraic contradiction.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：V120 部分保留；V121 目标已否定。

【证据与锚点】

- 类型：源码事实；日志观测；推断/未决
- 对话：visible messages 5396-5402。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`480f71c`。

【关联】

Teacher successor/delta identity, reliability shrinkage and V120 behavior anchor.

### CI-0419 — V122 weakened identity consumption, and the independent mainline was a new peripheral implementation rather than a mechanical extraction

【时间 / 版本】

V122 through initial independent mainline, reconstructed 2026-08-26.

【当时在解决什么问题】

Locate the ancestry of the all-object collapse that Schema20-25 later repaired.

【当时的 insight / hypothesis】

Removing identity/common information in the name of innovation can weaken a
consumer without reducing the Teacher target itself. A collapse visible at batch
20 and unchanged through epoch 8 points to initialization/topology rather than
late-training drift.

【具体做法】

Compared V122 with V120 metrics, then inspected independent-mainline commit
`91a4237` and its first complete run.

【观察到的结果】

V122 kept Teacher semantic RMS near V120 (`0.382` versus `0.3805`) but worsened
G object cosine (`0.713`), S interval/temporal variation (`0.0315/0.0060`) and
W2 object cosine (`0.592`), with best physical RMSE `0.08910`. The independent
mainline added 18,758 lines across 59 files and reimplemented the full
`G→S→Teacher→W→P1/P2/P3→typed ingress` periphery. G and W object-pair cosine were
already exactly `1.0` by batch 20 and remained so at epoch 8; best physical RMSE
was only `0.09107`.

【当时的解释】

V122's primary regression was identity/consumer suppression, not the later
Teacher-bandwidth collapse. The independent mainline then enlarged that problem
by redesigning, rather than faithfully extracting, the entire connection layer.

【后来怎么看（截至冻结终点）】

Retained. Schema20-25 are therefore better understood as recovery engineering,
not ordinary feature development.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留为谱系事实。

【证据与锚点】

- 类型：源码事实；日志观测；历史检索
- 对话：visible messages 5398-5402。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`91a4237`。
- 直接量值/表达式（卡内原文）：`G→S→Teacher→W→P1/P2/P3→typed ingress`。

【关联】

Initialization collapse, independent-mainline ancestry and Schema20-25 recovery.

### CI-0420 — Schema25 was a useful replay root and performance island, not final architecture truth

【时间 / 版本】

Schema24-fix through Schema26, replayed 2026-08-26.

【当时在解决什么问题】

Choose a historical point from which current failures could be replayed without
discarding later correct repairs.

【当时的 insight / hypothesis】

A replay root needs sufficiently good behavior and recoverable provenance, but
need not be structurally ideal. Calling it a root does not authorize whole-version
rollback.

【具体做法】

Cross-checked the supplied V120-to-V25 replay document with source and logs,
including performance, runtime, memory and top-path diagnostics.

【观察到的结果】

Schema25 achieved best/final physical RMSE `0.07724/0.07887`, about 11.65 GiB and
1.85 s/batch, while preserving restored bottom, query-first N=49 static P1, 24
queries/four glimpses/3×3 microgrid, exact Teacher algebra and S typed ownership.
It still had a 6.3× dynamic/static P1 ratio, weak geometry under type competition,
W adjacent cosine near `0.909`, duplicate P3 factual/effect aliases and a joint
lane×basis+null competition. Its log also lacked an action-normalizer fingerprint.

【当时的解释】

Schema25 was the clearest recovery island and a legitimate source-replay root,
but neither its apparent performance lead over V120 nor all of its structural
choices could be declared controlled truth.

【后来怎么看（截至冻结终点）】

Retained with the normalizer/provenance limitation.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；禁止整体回退。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；推断/未决
- 对话：visible messages 5407-5414。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。

【关联】

Replay roots, V25/V26 performance anchor and experiment fingerprinting.

### CI-0421 — Historical recovery should salvage ownership semantics, not entire commits

【时间 / 版本】

V121/V122/Schema35/Schema37 retrospective, 2026-08-26.

【当时在解决什么问题】

Determine which apparently good historical P2/P3 ideas could actually be
restored into a later graph.

【当时的 insight / hypothesis】

The reusable unit is often a data-ownership boundary rather than source code.
A historical implementation can contain a correct decomposition followed by an
incorrect fusion, so cherry-picking the whole commit recreates both.

【具体做法】

Classified candidate recovery items into mechanically reusable shapes, semantic
boundaries to re-express in current code and implementations that must not be
recovered.

【观察到的结果】

- V121 correctly created independent semantic and geometry keys/posteriors/values,
  but then made them compete through a type softmax and had already removed the
  camera axis.
- V121 correctly removed duplicate factual/effect P3 lanes, but its remaining
  precision/temporal lanes could still re-encode full protected fact.
- V122 improved ownership by using `effect + interaction` as temporal consequence
  innovation; however, K-mean centering deleted legal common detail and learned
  null could become another precision shortcut.
- Schema35's static/dynamic P1 type split and Schema37's independent per-lane
  4-basis+zero-null readers were worth retaining.

【当时的解释】

The safe composite was: independent complementary P2 types with camera retained;
protected consequence exactly once and outside null; P3 precision/temporal/state
change only when each has a unique source; independent optional-lane nulls; no
semantic-vs-geometry softmax or fabricated K-specific precision axis.

【后来怎么看（截至冻结终点）】

Retained as a recovery method, not authorization to implement the composite.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：设计语义部分保留；历史实现不可整搬。

【证据与锚点】

- 类型：源码事实；过程/决策记录；推断/未决
- 对话：visible messages 5403-5414。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。

【关联】

Complementary evidence, unique owner, precision private source and per-lane null.

### CI-0422 — V25's low-dustbin association was a counterfactual Teacher backend, not proof that flat softmax should replace partial OT

【时间 / 版本】

Schema25 versus Schema39 Teacher replay, 2026-08-26.

【当时在解决什么问题】

Assess whether the historical Teacher could recover temporal target bandwidth
without guessing a new association rule.

【当时的 insight / hypothesis】

Large behavioral differences justify an equal-input counterfactual backend, but
not immediate replacement: flat row-wise matching and partial OT impose
different cross-object occupancy assumptions.

【具体做法】

Compared V25's per-current-object flat future-camera/cell+null softmax with the
current background-subtracted partial OT and their logged outputs.

【观察到的结果】

V25 versus Schema39 showed dustbin around `0.042` versus `0.473`, Teacher semantic
delta around `0.350` versus `0.206`, and interval variation around `0.134` versus
`0.053`. V25's method allowed multiple current objects to occupy the same future
cell, while partial OT constrained shared matching mass.

【当时的解释】

Candidate scoring, row normalization and single identity fallback from V25 were
worth implementing as an isolated, same-input comparison backend combined with
newer camera moments and neutral status semantics. The numbers did not prove
that partial OT should be deleted wholesale.

【后来怎么看（截至冻结终点）】

Retained as an unrun counterfactual and unresolved association question.

【冻结终点判定】

- 结论处置：待证
- 证据闭环：未闭环
- 说明：尚未验证。

【证据与锚点】

- 类型：因果干预；推断/未决
- 对话：visible messages 5413-5414。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。

【关联】

Teacher association, dustbin semantics and IC-06-style supervision uncertainty.

### CI-0423 — Schema37, 38 and 39 repaired three different layers, so their health gains and remaining behavior gap can coexist

【时间 / 版本】

Schema37-Schema39 replay, 2026-08-26.

【当时在解决什么问题】

Add the three newest versions to the historical ledger without flattening them
into one phase or judging them only by one metric name.

【当时的 insight / hypothesis】

Identity preservation, action-consumer reachability and correct terminal-axis
placement are separable layers. Closing one does not imply adequate upstream
target bandwidth or final task benefit.

【具体做法】

Recorded each version's source change, complete/partial log coverage, inherited
items and prohibited carryovers, using Schema36 only as their direct parent.

【观察到的结果】

Schema37 primarily preserved information identity/axes; Schema38 made W, S and
dynamic P1 reach action consumers; Schema39 separated P2 spatial selection from
P3 temporal termination and moved main dynamic precision into a protected
no-null carrier. Therefore Schema39 optional precision `0.0202` could not be
compared alone with older total precision; it had to be read with protected
precision `0.2312`. Teacher bandwidth, S-W temporal leverage and P3 ownership
still remained open.

【当时的解释】

The three versions were neither useless nor independently complete. Keeping the
correct layer from each was more accurate than discarding all three or restoring
one wholesale.

【后来怎么看（截至冻结终点）】

Retained in the architecture replay ledger; Schema39 log completeness remained
limited to seven full epochs at that moment.

【冻结终点判定】

- 结论处置：部分保留
- 证据闭环：部分闭环
- 说明：部分保留；组合闭环未完成。

【证据与锚点】

- 类型：日志观测；过程/决策记录；推断/未决
- 对话：visible messages 5423-5428。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。

【关联】

Metric ownership, layered closure and architecture replay.

### CI-0424 — Historical audit documents are evidence to criticize, not executable plans

【时间 / 版本】

Architecture-replay documentation, 2026-08-26.

【当时在解决什么问题】

Combine two supplied historical analyses into a durable reconstruction without
silently promoting their recommendations to current scope.

【当时的 insight / hypothesis】

An audit document can correctly identify ancestry while overstate a rollback
point or preserve formulas later superseded. Its claims need the same
fact/inference/recovery-boundary classification as a live diagnosis.

【具体做法】

Cross-checked both supplied documents against the local snapshots, commit diffs,
logs and current Schema39 consumers, then wrote `ARCHITECTURE_REPLAY_LEDGER.md`
and indexed it under auxiliary research.

【观察到的结果】

The resulting ledger classified V120 as a behavior anchor, Schema24-fix/V25/26
as a recovery island, V27 and V31 as distinct breaks, Schema32 as a useful
one-way W ownership repair, and Schema36-39 as correct axis/consumer repairs
that did not restore Teacher-to-time-terminal bandwidth. It separately listed
recoverable semantics and historical implementations that must not be copied.

【当时的解释】

This allowed future work to use history for multiple-angle diagnosis without
turning archaeology into an accidental implementation directive.

【后来怎么看（截至冻结终点）】

Retained. The conversation insight ledger now complements it by preserving the
reasoning and later reversals that a source replay alone cannot show.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 5407-5428。
- 段级工件：见 Segment 042 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`ARCHITECTURE_REPLAY_LEDGER.md`。

【关联】

Historical evidence discipline, auxiliary research and current-contract separation.

## Segment 042 synthesis

The deeper replay showed that V27 was not the origin of all problems. V120 had
measurable but non-fatal debt; V121 introduced a target identity contradiction;
V122 weakened S/object consumption while the Teacher magnitude remained similar;
and the independent mainline replaced the entire peripheral graph and collapsed
G/W identity almost immediately. Schema20-25 were consequently recovery work,
with Schema25 forming a valuable performance/replay island rather than a final
truth.

The historical designs were then decomposed below commit granularity. Independent
P2 semantic/geometry evidence, unique protected consequence, V122's
consequence-innovation operand, static/dynamic P1 ownership and per-lane zero-null
readers were reusable semantics. Type softmax, lost camera axes, duplicate P3
lanes, K-centering and unrestricted learned null were not. V25's low-dustbin
Teacher became an equal-input counterfactual candidate rather than a mandated
rollback. Finally, Schema37-39 were recorded as three distinct layers of partial
closure, preventing metric names from hiding that their information owners had
changed.

## Segment 043 — Recoverable conversation memory and the insight-ledger evidence method

### Segment boundary

```text
Date:                 2026-08-26 to 2026-08-28
Visible messages:     5429-5493
Starts with:          whether context compaction caused repeated implementation drift
Ends with:            exact-history check of whether B-spline was ever a real proposal
Primary historical artifacts:
  Codex thread-history pages
  C:\Users\ASUS\.codex\sessions\2026\07\18\rollout-*.jsonl
  C:\Users\ASUS\.codex\thread_history_1.sqlite
  docs/research/auxiliary/CONVERSATION_INSIGHT_LEDGER.md
```

### CI-0425 — Context compaction amplified drift, but incomplete source review converted drift into code defects

【时间 / 版本】

Whole-task process retrospective, 2026-08-26.

【当时在解决什么问题】

Assess whether repeated version confusion and half-implemented repairs were
caused by the conversation becoming extremely long.

【当时的 insight / hypothesis】

Compaction can lose precise version boundaries, tensor ownership and old
negative constraints from the active reasoning context. It is an important risk
amplifier, but it cannot explain deterministic dtype/shape/startup errors or
replace the obligation to reread the actual subsystem.

【具体做法】

Separated limitations of the automatically supplied context from engineering
actions that were always locally verifiable. Reconstructed the causal chain as
`long context/compaction → easier semantic drift → missing persistent correction
→ broad or incomplete source review → real implementation error`.

【观察到的结果】

Compaction plausibly blurred V120 variants, solved versus open problems,
protected versus optional carriers and old "do not rebuild" boundaries. It did
not excuse BF16/FP32 errors, bad `expand`, 6-GiB broadcasts, missing attributes,
script substitution bugs, untested activation boundaries or rebuilding stable
P1/bottom paths.

【当时的解释】

The controllable failure was relying on compressed memory instead of forcing
every change through current source, persistent ledgers and a complete
producer/consumer/backward/runtime review.

【后来怎么看（截至冻结终点）】

Retained. It directly motivated the workspace memory and mandatory subsystem-
familiarity rule now supplied in `AGENTS.md`.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；测试/运行观测
- 对话：visible messages 5429-5431。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`AGENTS.md`。

【关联】

External memory, source-first work and release closure.

### CI-0426 — The active model context is not the complete task archive; the latter can be deliberately replayed

【时间 / 版本】

Codex task-history investigation, 2026-08-26.

【当时在解决什么问题】

Determine whether old assistant messages and tool events were irretrievably
lost after compaction.

【当时的 insight / hypothesis】

"Not automatically present in this inference context" is different from "not
persisted." Claims about inaccessible history must be tested against the app's
thread API and local event storage.

【具体做法】

Queried paginated thread history and inspected local Codex storage, locating the
task id, SQLite history/state stores, session index, attachment directory and
the continuously written rollout JSONL.

【观察到的结果】

The history API returned old user/assistant messages, commands, edits, reasoning
summaries and compaction markers with `hasMore=true`. The local rollout retained
the task event stream. Full hidden chain-of-thought was not recoverable, and some
terminal outputs could be truncated, but visible messages and principal tool
events could be systematically replayed.

【当时的解释】

The earlier statement that the complete history could not be seen was too
absolute. The correct boundary was `current context ≠ complete persisted task
archive`; deliberate retrieval was available and should have been used for
archaeology.

【后来怎么看（截至冻结终点）】

Retained and operationally demonstrated by the extraction of Segments 001-043.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留；早期过强表述已纠正。

【证据与锚点】

- 类型：源码事实；测试/运行观测；过程/决策记录；历史检索；推断/未决
- 对话：visible messages 5432-5439。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。

【关联】

Compaction, task replay, evidence provenance and this ledger.

### CI-0427 — The raw rollout is an event stream, not a ready-made pure conversation file

【时间 / 版本】

Conversation-archive inspection, 2026-08-26.

【当时在解决什么问题】

Identify which persisted file could serve as the complete human-readable
conversation source.

【当时的 insight / hypothesis】

Raw preservation and readable evidence projection should be separate artifacts.
Filtering must retain message identity and attachment provenance while excluding
system/runtime noise from the human narrative.

【具体做法】

Classified the local rollout, history/state databases, session index and
attachments by content and inspected the current task's approximately 789-MB
rollout file.

【观察到的结果】

No existing file was only user/assistant prose. The rollout mixed visible
messages with system/developer context, tool calls, terminal output, edits and
compaction events; the SQLite history was closest to the UI but was not a text
document. A clean export would need to select user messages and assistant
commentary/final text while retaining timestamps, visible IDs and attachment
references.

【当时的解释】

The event stream remained the authoritative raw source, while this ledger became
an interpreted, evidence-linked projection rather than a replacement archive.

【后来怎么看（截至冻结终点）】

Retained. This extraction indexed non-empty visible messages while explicitly
ignoring injected environment/plugin records as insights.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；过程/决策记录；历史检索
- 对话：visible messages 5444-5449。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。

【关联】

Raw archive, clean conversation export and visible-message cursor semantics.

### CI-0428 — Historical insights need chronology, implementation identity, later correction and evidence class in the same card

【时间 / 版本】

Insight-ledger design and initial extraction, 2026-08-26 to 2026-08-27.

【当时在解决什么问题】

Record years of architecture discussion without flattening hypotheses, plans,
actual code, experiments and later reversals into one current-sounding summary.

【当时的 insight / hypothesis】

Chronology preserves why a decision looked reasonable; stable IDs and later-
view/status fields preserve how evidence changed it. Plan and realization must
be separate because many failures came from a sound intention implemented only
halfway.

【具体做法】

Adopted the user's card template—time/version, problem, hypothesis, concrete
method, observation, contemporary explanation, later view, status and links—
plus stable IDs, actual commit/schema/run identity, source/log/inference evidence
classes, involved dataflow boundary and current causal category. Extraction used
small visible-message ranges, full rereads for long turns, natural topic
boundaries, immediate writes and card/status/evidence consistency checks.

【观察到的结果】

The method preserved reversals such as blanket `transition_detach` becoming
selective causal isolation, 22/22 structural tests being disproved by the first
full epoch, the near-zero-flow allegation being narrowed by a dataset probe and
V97's nominal role split being corrected by its unrealized raw-flow details.

【当时的解释】

Recording only final conclusions would erase the counterevidence that makes
those conclusions reliable. Recording only the raw transcript would leave
future agents unable to distinguish adopted design from abandoned discussion.

【后来怎么看（截至冻结终点）】

Retained; the completed ledger itself is the validation artifact.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留并执行。

【证据与锚点】

- 类型：源码事实；日志观测；测试/运行观测；数据探针；过程/决策记录；推断/未决
- 对话：visible messages 5440-5443 and 5447-5487。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。
- 直接标识（卡内原文）：`transition_detach`。

【关联】

Insight lifecycle, historical replay and evidence discipline.

### CI-0429 — Candidate-count, capacity-group and batch-axis "8/4" quantities were historically distinct

【时间 / 版本】

V94 retrospective clarification, 2026-08-27.

【当时在解决什么问题】

Clarify whether the historical `8×4` crash implied that four execution
candidates were too few.

【当时的 insight / hypothesis】

Identical integers do not imply a shared mechanism. Axis identity and coordinate
semantics matter more than numeric coincidence.

【具体做法】

Separated `batch=8 × candidates=4`, `rank=32/groups=4` and the later
`3 blocks × 2 dwell=6` table. Reconstructed both the right-aligned broadcast and
the meaning of each candidate row before/after a hard state transition.

【观察到的结果】

The crash was caused by one extra singleton dimension, not candidate capacity.
Four rows could express current/next by dwell 1/2, but their identities changed
with the hard current block, warmup used a different graph, and the selector
often became a uniform average. Replacing them with six globally fixed rows
solved coordinate instability; it did not eliminate the later soft/hard dwell
gap. Separately, four rank groups were a coarse capacity quantization and were
replaced for a different reason.

【当时的解释】

The durable fix was stable candidate identity across warmup/train/deploy and no
hard recurrent state in training, not "increase 4 to 6" as a capacity heuristic.

【后来怎么看（截至冻结终点）】

Retained as a clarification of CI-0011 and CI-0015.

【冻结终点判定】

- 结论处置：保留
- 证据闭环：已闭环
- 说明：保留。

【证据与锚点】

- 类型：日志观测
- 对话：visible messages 5464-5467。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。
- 直接量值/表达式（卡内原文）：`rank=32/groups=4`。

【关联】

CI-0011, CI-0015, axis identity and schedule-boundary testing.

### CI-0430 — The V95 near-zero-flow suspicion was narrowed by a matched data probe before it changed V96's design

【时间 / 版本】

V95-V96 retrospective clarification, 2026-08-27.

【当时在解决什么问题】

Recover the complete hypothesis→objection→probe→revision chain rather than only
the final V96 architecture.

【当时的 insight / hypothesis】

An internal RMS that resembles a known shortcut is a hypothesis trigger, not a
verdict. Units, support, data prior and matched interventions must precede
forcing the quantity away from its observed scale.

【具体做法】

Re-read the original V95 flow metrics, the user's objection that the frames were
genuinely similar, the exact four-frame/8×8/all-patch logger semantics and the
same-distribution DINO data probe. Connected those findings to the later removal
of the single long-range Stage token and the identity-centred late address read.

【观察到的结果】

Roughly 20% moving support at 0.08 patch displacement can produce the observed
global mean near `0.016-0.017`; short motion was sparse while long-horizon local
motion accumulated. The probe invalidated the strong collapse accusation but
did not prove learned flow useful without learned/zero/shuffled checkpoint
interventions. It also showed untrained global soft correlation could be worse
than identity and too diffuse.

【当时的解释】

The design emphasis moved from forcing nonzero flow to preserving spatial
long-horizon evidence and using bounded identity-centred local address priors.
V96 removed both the separate training Stage1 and the one-vector `t+48` Stage
representation, while preserving Flow's local-address role. Its remaining
DINO-only precision limitation later motivated V97/V98 raw detail.

【后来怎么看（截至冻结终点）】

Retained as a detailed cross-reference to CI-0063, CI-0065 and CI-0069; no new
claim that the missing checkpoint intervention had been run.

【冻结终点判定】

- 结论处置：已吸收
- 证据闭环：部分闭环
- 说明：被后续方案吸收；因果收益仍需匹配干预证明。

【证据与锚点】

- 类型：因果干预；数据探针；过程/决策记录；方案/设计记录；推断/未决
- 对话：visible messages 5488-5490。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。

【关联】

CI-0063, CI-0065, CI-0069, data-normalized scale and V96 late bottleneck.

### CI-0431 — B-spline was only a recording-template example in this task history

【时间 / 版本】

Whole-task exact-history search, 2026-08-28.

【当时在解决什么问题】

Determine whether ClearVLA had historically discussed or implemented a
B-spline action representation.

【当时的 insight / hypothesis】

An illustrative example in a documentation template must not be promoted into
project ancestry without a proposal, source change, log or experiment.

【具体做法】

Searched the complete available task history for `B-spline`, `spline`, Chinese
B-spline terms and related representations, while distinguishing the user's
template example from substantive architecture turns.

【观察到的结果】

The only matches were the 2026-08-26 insight-card example and the assistant's
expansion of that example. No earlier design discussion, version, implementation,
source change or experimental result was found.

【当时的解释】

Within this task, B-spline was hypothetical documentation content, not an
attempted ClearVLA architecture. If such work exists elsewhere, it requires an
independent artifact; it cannot be reconstructed from these two mentions.

【后来怎么看（截至冻结终点）】

Retained as a terminology and ancestry guard. `B-spline` means the mathematical
continuous curve defined by knots and control points, not a policy backbone or
the whole velocity call.

【冻结终点判定】

- 结论处置：已否定
- 证据闭环：已闭环
- 说明：非历史方案；禁止误记。

【证据与锚点】

- 类型：过程/决策记录；历史检索
- 对话：visible messages 5442-5443 and 5491-5493。
- 段级工件：见 Segment 043 boundary 的 `Primary historical artifacts`。

【关联】

Historical evidence classification and the B-spline terminology guard; no MIP
relationship was established.

## Segment 043 synthesis

The last segment turned the conversation itself into a verifiable research
source. Long-context compaction was recognized as a real drift amplifier, but
not an excuse for errors that complete source review and production-path tests
should have caught. The apparently inaccessible early discussion was then found
in paginated task history and a local mixed event stream, establishing a clean
boundary between active model context, persisted archive and interpreted ledger.

The adopted card format preserves contemporary reasoning and later correction,
actual implementation identity, evidence class and causal status. Incremental
range reads and immediate validation kept extraction from repeating the same
memory problem it was intended to solve. Two historical clarifications also
demonstrated the method: the V94 `8×4` event was a batch/candidate broadcast error
plus an unstable candidate coordinate—not evidence that four was inherently too
small—and the V95 small-flow suspicion was narrowed by units and a dataset probe
before motivating a spatial V96 design. Finally, an exact search established
that B-spline appeared only as a template example, preventing a hypothetical
term from being rewritten as project ancestry.

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
