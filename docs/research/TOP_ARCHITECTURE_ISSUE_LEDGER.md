# ClearVLA 顶层源码—论文问题账本

状态：审查对象为当前 `object_intent_dynamics_323` 工作树，不按 Vxxx 名称推断实现
更新：2026-08-08
范围：Pre-G 之后的 G / S / Teacher / W / P1 / P2 / P3、相邻损失、静态缓存和唯一 bottom ingress
不在范围：Evidence MMDiT、CVAE、workspace、execution controller 内部能力重做

当前执行图和禁止输入仍以
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)
为准。本文件记录当前源码已确认的问题、日志证据、论文能提供的机制和下一次主线修改边界；它不是新的版本契约。

证据标签：

- **[源码]**：当前工作树可以直接证明；
- **[日志]**：三个完整 V120 及已有冻结探针支持；
- **[推断]**：源码和日志共同支持，但还不能单独证明因果量；
- **[方向]**：下一版候选设计，不冒充已经实现。

## 1. 已锁定的当前证据

- 修复 binder 且移除不可识别 completion、改用零中心 state-change 的完整 V120，最好 action RMSE 约 `0.07931`（E7），E8 为 `0.08145`；E8 tail 约 `0.09590`、gripper 约 `0.14979`、event-head F1 约 `0.131`。completion 修复有收益，但没有解决对象—未来—动作的主要错位。
- W 并非断线：W1/W2、effect heads、P2、consequence 均有有限梯度，四区间 target-normalized error 也会下降。
- W 仍显著过平滑：prediction adjacent cosine 约 `0.915`，teacher 约 `0.856`；prediction interval variation 约 `0.0965`，teacher 约 `0.140`。也就是预测变化量比目标少约三成，不能解释成“teacher 本来就完全相同”。
- transport normalized error 已从旧 completion 路径的约 `0.94–1.01` 改善到约 `0.61–0.65`；但 P2 最终 type mass 约为 semantic `0.285` / geometry `0.114` / status `0.602`。这不是目标分布，却暴露出几何改善没有转化成主要在线 value 所有权。
- P1 spatial variation、detail output、seed reliability 和 late-detail update 随 action RMSE 改善同步增强。因此当前收益不能直接归功于 W，也不能用削弱 P1 来“强迫 W 有用”。
- 当前 `grab_pen_single` 只有单一语言条件。高 goal-attention entropy 在这份数据上主要说明语言差异不可识别，不能单独证明接线错误；同样也不能据此宣称语言组织已经学好。

## 2. 论文边界：能迁移什么，不能照搬什么

2026 年 8 月存在两篇同名 `Faster-WAM`，本账本始终分开引用。

| 工作 | 论文中真正有效的机制 | 对当前主线的启示 | 明确不照搬 |
|---|---|---|---|
| [Fast-WAM](https://arxiv.org/abs/2603.16666) | 训练期 future-video co-training 塑造在线 action 表征；action 不读取训练期未来 token；部署删除未来分支 | Teacher/recognizer 只能改变 target/loss；保留 current-only policy 基线；把 training-only W 与 cached online W 做受控比较 | 不恢复像素视频生成，不把 future teacher 塞进部署 value |
| [Faster-WAM：Efficient Inference-Time Future Conditioning](https://arxiv.org/abs/2608.04404) | 一次构建并缓存 future-aware K/V；只在少量 action stage 交互；同一深度区间的 K/V 成对融合 | G/S/W/P1 保持一次缓存；W 只通过一个明确 future dock 进入 action；K/V 共享选择权重 | 不把 W 注入每个 block，不做 hard gate/usage quota；论文的“interval”是深度区间，不能拿来平均本项目四个时间区间 |
| [Faster-WAM：Dock of Transformer](https://arxiv.org/abs/2608.02365) | world backbone 与 action head 可不同深度；显式 dock 汇聚多层 K/V；跨模态位置基底先重对齐 | 当前 local chart、global K object、action-time query 必须在显式 dock 内对齐；world/action 宽度和深度不必一一对应 | 当前 P2 没有 3D/1D RoPE，不生造 RoPE 修复；不据此把已有 bottom action tower 删成单层 |
| [MV-WAM](https://arxiv.org/abs/2606.21088) | 视觉、动作、价值属于异质流形；使用 modality-specific experts/objectives 和单向信息流；value 依赖真实 return | semantic/geometry/status/action 需要不同 key/value/目标几何；Teacher→W→P 必须单向且可追溯 | 没有 return/失败/rollback 数据时不引入 value、completion 或 progress；不把 action flow-matching 改为论文的其他动作目标 |

论文共同支持的不是“再堆一个 world block”，而是三件事：

1. 训练期未来证据与部署值严格隔离；
2. 世界表征和动作表征通过少量、显式、可对齐的 dock 交换；
3. 不同语义流保留各自的 key/value 和目标，而不是共用一个 posterior 后再改名字。

## 3. 当前真实轴变换

```text
local G hypotheses       [B,C,8,8,M,*]
  -> global binder
ObjectFactSet            [B,K,*], K=4
candidate_assignment     [B,K,C,8,8,M]
object_to_chart          [B,K,C,8,8]
  -> S
interval_queries         [B,4,H]
temporal_queries         [B,24,H]
  -> W
FutureObjectDynamics     [B,4,K,*]

旧 local G lattice ---------------------------------> P1 -> [B,T,Q,H]
FutureObjectDynamics + action/S query --------------> P2 -> [B,T,Q,H]
P1 + P2 -------------------------------------------------> consequence/P3
P3 typed values         [B,5,T,Q,H]
protected_base          [B,T,Q,H]
  -> bottom protected-detail reader
  -> controlled-dynamics context（当前重复入口）
```

最关键的事实是：`candidate_assignment` 已经保留了 global K object 到 local `C×8×8×M` 的软反查，但 P1 没有使用它；P1 先在旧 local lattice 上聚合，W/P2 则在 global K 上组织。两套身份只在聚合后的 hidden addition 中相遇。

## 4. 从顶向下源码账本

### 4.1 Local G → DenseFactChart

源码位置：

- `clearvla/policy/object_intent_dynamics_323/grounding.py:30-74`
- `clearvla/policy/object_intent_dynamics_323/types.py:61-108`

**[源码] 当前做法**

- `dino_content` 由 `semantic_owner_probs × validity` 聚合（`grounding.py:34-38`）。
- local-M 条件先验是 `sqrt(semantic × geometry)`（`43-55`）；appearance 不决定物理存在/null，这是正确边界。
- semantic、appearance、geometry、coordinate、support、validity 和 transport prior 在 DenseFactChart 中仍分别保存，没有在这里被真正删掉。

**[日志]** `object_grounding_local_prior_entropy` 长期接近 `0.999`，说明这个 local-M 先验在当前数据上近似均匀；它不等于 global object binder 坍缩。

**问题 `G-LOCAL-PRIOR`（P1）**

- **[推断]** 物理先验几乎不区分 local M，后续 global binder 主要依赖 learned candidate token，而不是已有 typed localization。appearance 虽然保留为特征，却没有自己的验证 posterior。
- 这会向下传播为：global object 可能很尖锐，但“为什么是这个 object”难以分解成 semantic/appearance/geometry 三种证据。

**[方向]** 保留一个共同的物理 assignment/null；appearance 仍不进入存在性，但必须保留为 object 内部的 verification evidence，用于 P1 精细候选校验。禁止通过 entropy/diversity loss 强制先验不均匀。

### 4.2 DenseFactChart → global K object binder

源码位置：

- `grounding.py:145-183`：候选 token 与 K+null competition；
- `grounding.py:216-254`：G3 bounded residual 与统一聚合；
- `grounding.py:262-332`：existence/validity/object-to-chart/reconstruction；
- `grounding.py:335-374`：当前诊断。

**[源码] 当前做法**

- candidate key 同时包含 content、semantic、appearance、geometry 和 coordinate（`145-151`）。
- 只有一套 K+null assignment；随后同一个 `read` 聚合 content、semantic、appearance、geometry、coordinate 和 transport（`245-254`）。
- G3 是 `log(parent posterior) + bounded residual`（`216-243`），零 residual 精确继承 G2，这一数值边界是健康的。
- `candidate_assignment [B,K,C,8,8,M]` 和 `object_to_chart [B,K,C,8,8]` 都已经存在；前者保留 local M，后者已对 M 求和。
- `object_content_pair_cosine` 只比较聚合后的 DINO content（`369-374`），没有测 typed owner overlap 或 object-to-chart overlap。

**[日志]** global candidate-owner entropy 约 `0.22`，而 object content pair cosine 约 `0.5`。因此可以说 binder 的 global assignment 较尖锐、内容并非完全相同；不能说 typed ownership 已经健康，也不能说四槽全部坍缩。

**问题 `G-TYPED-IDENTITY`（P0）**

- **[源码]** semantic/appearance/geometry 最终共用一套 object posterior。它们有不同 value，却没有不同的“验证这个 object 的证据分布”。
- **[源码]** 日志只测 content cosine，当前所谓“槽同质”无法区分：内容相似、空间支撑重叠、还是 typed evidence 共用 assignment。
- **[推断]** 这解释了为什么下游 S/W 虽然收到三个命名字段，却仍可能表现为同一组 object 的不同线性投影。

**[方向]**

- 继续使用一个共同物理 K+null assignment，避免三套 object 身份互相漂移；
- 在共同 assignment 上保留 semantic support、appearance verification、geometry support 三套 bounded residual/posterior；
- 直接复用现有 `candidate_assignment` 作为 global-object→local-candidate dock，不新造第二套 object finder；
- 新诊断分别记录 object-to-chart overlap、typed posterior overlap 和 content cosine，三者不能再用一个 pair cosine 代替。

这与 MV-WAM 的异质专家思想一致，但不复制三套完整 backbone；与 DoT 一致的重点是明确坐标和接口，不是增加深度。

### 4.3 Online S：StatelessObjectIntentOrganizer

源码位置：

- `clearvla/policy/object_intent_dynamics_323/intent.py:107-392`
- `clearvla/policy/trunk.py:11261-11355`
- `clearvla/policy/trunk.py:5308-5384,5453-5469`

**[源码] 当前做法**

- 四个 goal queries 读取完整 T5；history 保留顺序、状态差分和相对历史位置；四个 interval identity 分别读取 goal/history/object/typed object；typed AttnRes 在三种 named innovation 形成后才路由。这部分没有回到伪 phase/progress。
- `interval_queries` 已经包含 goal、history、object 和 typed innovations（`intent.py:269-291`）。
- 但 object 主线随后又把 `protected_goal_set.mean(dim=1)` 和最后一个 history token 分别扩成四份（`trunk.py:11346-11355`）。
- P1 query 最终相加 `phase_query_delta + condition_query_delta + history_query_delta`（`5453-5469`）。也就是同一 goal/history 已经在 canonical S interval 中出现一次，又通过两个 ancestry alias 再出现一次。

**问题 `S-P1-ALIAS-DUPLICATION`（P0）**

- **[源码]** P1 的合法 S query 被重复成三个相关入口。这个不是“信息更丰富”，而是同一来源的多次投影；它给 S→P1 current-only 路径更多幅度和优化机会，而 W 只在 P2 出现一次。
- **[推断]** 这是当前 P1/S 旁路容易主导、W 效益难归因的结构性原因之一。

**[方向]** object 主线的 P1 只接收 canonical `interval_queries`；删除 object capability 下的 mean-goal 与 last-history query alias。Goal/history 仍完整保留在 S 内，不是删除语言和历史。P1 的 S 输入只能改变地址/query，不能制造 RGB/detail value。

**非问题边界**

- 单 T5 数据上的 goal-attention entropy 接近 1 不是独立错误；不加 entropy target。
- history 中 `[-1,0]` 的相对位置编码是顺序编码，不是 episode progress。

### 4.4 Training-only recognizer 与 CoarseActionIntent

源码位置：

- `intent.py:395-480`：FuturePlanRecognizer；
- `intent.py:484-560`：CoarseActionIntent；
- `trunk.py:11308-11337`：online S matching 与训练 target。

**[源码] 当前做法**

- recognizer 对每个 interval 的 future action/state 直接做时间均值（`intent.py:431-436`）。
- teacher semantic effect 又对 K objects 做 validity-weighted 均值（`447-451`）。
- action/state/effect 三个均值相加成一个 hidden token，online S 用 SmoothL1 拟合它（`trunk.py:11314-11320`）。
- CoarseActionIntent 也用每区间 future action 均值作训练 target；在线值只来自当前 S/G/history，且是 W 的唯一 action 输入，这一 teacher/deploy 隔离是正确的。

**问题 `S-AVERAGED-TEACHER`（P1）**

- **[源码]** recognizer 在监督 S 前先丢掉区间内顺序和 global K object 身份。
- **[推断]** 这直接给 S 一个“学平均摘要”的优化压力，并把这种公共方向继续送入 W；它与 W1/W2 相似性高是相关机制，但不能仅凭源码声称是唯一原因。

**[方向]**

- 保留 training-only recognizer，但输出 factorized target：interval action intent、state endpoint/delta、`[I,K]` object effect K/V；
- online S 通过 cross-read/reconstruction 对齐这些 target，不把所有分量先相加成一个 hidden label；
- recognizer target 只塑造在线 S，仍不进入部署 value；
- 不加入 scalar progress、phase label、completion 或无 return 的 value。

这采用 Fast-WAM 的“训练期未来分支塑造在线表征”边界，同时避免把 MV-WAM 的 return/value 机制生搬到无失败标签的数据上。

### 4.5 Teacher-G：current object → future supports

源码位置：`clearvla/policy/object_intent_dynamics_323/teacher.py:14-286`

**[源码] 当前做法**

- Teacher 全程 no-grad/FP32；完整 DINO value 被保留，只有 association key 降到固定 64 维（`92-98,151-158`）。
- association key 只来自 `facts.content`；geometry、flow、camera prior 分别加入，但 semantic/appearance/geometry typed object fields 没有独立 association key（`92-145`）。
- 同相机内仍允许全局软匹配并有 null，非等像素、遮挡和零流都是合法的，这部分是健康的。
- 每个 `4–8 / 8–16 / 16–32 / 32–48` 区间内部对 supports 均匀平均（`204-238`）；persistence 例外，使用可见性的几何均值。

**问题 `TEACHER-TYPED-SMOOTHING`（P1）**

- **[源码]** 当前 Teacher 是“DINO content + coordinate/flow prior aligned”，不是完整 typed-G aligned。
- **[源码]** 区间内 uniform mean 会抹去 start/end/变化方向；teacher adjacent cosine 已约 `0.856`，W 再平滑到约 `0.915`。

**[方向]**

- semantic compatibility、appearance verification、geometry/flow transport 分开产出 key/logit，再在一个 null-capable association 中合成；
- 每个时间区间保留有序的 start/mean/end 或小型 K/V summary，K/V 使用同一 support 权重；不把四个真实时间区间互相平均；
- 不人为压低 cosine；如果样本确实静态，预测和 teacher 允许相似；
- future raw image 仍不是运行前提，Teacher 继续每训练 batch 一次、部署零次。

Faster-WAM 的 Interval KV-Fusion 可借鉴“成对 K/V 与多层信息保留”，但其 interval 是网络深度，不是本项目的时间区间；这里只迁移 K/V 组织原则。

### 4.6 W1/W2：FutureObjectDynamics

源码位置：`clearvla/policy/object_intent_dynamics_323/dynamics.py:80-363`

**[源码] 当前做法**

- W base 同时加入 object content、S interval、CoarseActionIntent、interval identity 和 protected goal second landing（`145-194`）；semantic/appearance/geometry 是三种 typed value。
- W1 负责近两个区间，W2 读取完整两-token W1 sequence 后预测远两个区间（`272-363`），没有再做 mean-pool，这部分正确。
- W1/W2 使用不同 interval blocks，但四区间共用同一组 delta/transport/covariance/visibility/persistence/uncertainty heads（`128-143,196-243`）。
- `future_address` 被生成并序列化；当前 P2 不读取该分布，只重新用 `object_coordinates + transport_mean` 算一个中心分数。

**问题 `W-COMMON-MODE`（P1）**

- **[源码]** S interval、CoarseActionIntent 和 protected goal 是高度相关的当前条件；goal/object/history 在 S 与 CoarseActionIntent 中已经出现，W 又重新读取 protected goal 和 typed objects。
- **[源码]** shared decoder heads 给 W1/W2 一个天然公共输出基底。
- **[日志]** W 不是断线，但 prediction variation 低于 target，符合“共享条件和共享头让区间差异被平滑”的表现。

**[方向]**

- W 输入改为一个 protected current-object base，加四类零中心 innovation：interval intent、coarse action、typed semantic/appearance/geometry、W1→W2 causal update；同一来源只出现一次；
- 保留共享 backbone，但为 W1/W2 或四 interval 增加轻量、显式的 typed K/V adapter/head identity，而不是再增加 block 数；
- W 输出成为一次缓存的 `ObjectFutureKVHierarchy`：semantic K/V、geometry K/V、status/calibration；同一监督对象仍是 P2 消费对象；
- CoarseActionIntent 不读 noisy ODE action，因此 W 仍可在五步部署中构建一次。不能把这种“静态在线 action intent”误写成 Faster-WAM 的完全 action-independent future branch。

### 4.7 P1：当前高分辨率事实读取

源码位置：

- `clearvla/policy/trunk.py:10090-10220`；
- `clearvla/policy/trunk.py:2481-2550,5308-5507`。

**[源码] 当前做法**

- P1 只执行一次高分辨率读取，RGB/detail value 和 local `C×8×8×M×micro` 精度仍被保留；这部分不能削弱。
- object 主线的 P1 从 `progressive_address_state` 的 ancestral local G3 lattice 读取，输入中没有 `ObjectFactSet.candidate_assignment` 或 global K object id。
- `policy_factual_rollout=world_detail_entry_rollout`，因此 P1 使用 protected current G3 chart，而不是 W working hidden；这一 teacher/W 隔离是正确的。
- S/goal/history 只进入 query/key 路径，value 仍来自观察；但如 4.3 所述，S 来源被重复三次。

**问题 `P1-W-OBJECT-BASIS`（P0，当前最重要）**

- **[源码]** P1 在 local hypothesis 空间先聚合，W/P2 在 global K object 空间组织。它们没有共同 object posterior，也没有共同坐标变换。
- **[推断]** 因此 W 学到“某个 global object 的未来 effect”时，P2 只能把聚合后的 effect hidden 加到已经失去 object identity 的 P1 fact 上；这正是 W 有 loss/gradient、却难稳定帮助精细 action 的结构解释。

**[方向] Object–Chart Future Dock**

1. 复用现有 `candidate_assignment [B,K,C,8,8,M]`，不另建 object detector；
2. action/S query 先在 K+null 上形成软 object posterior；
3. 在被选 object 内，用 `candidate_assignment` 约束 local cell/M，再执行现有 3×3 micro RGB/detail 软读取；
4. 流式计算 `fact_by_object [B,T,Q,K,H]` 和聚合后的 `fact [B,T,Q,H]`，不物化完整巨大 posterior；
5. P2 在同一个 K posterior/坐标基底读取 W effect。

这迁移 DoT 的显式 docking 与位置基底重对齐；它不是 hard crop，也不降低 raw/DINO 分辨率，更不恢复第二次视觉读取。

### 4.8 P2：FutureObjectDynamics → action query

源码位置：`clearvla/policy/object_intent_dynamics_323/compiler.py:79-225`

**[源码] 当前做法**

- 一个 interval×object posterior 由 semantic-delta content score、S interval score 和未来中心 coordinate score合成（`132-164`）。
- semantic delta、transport、visibility/persistence 三类 value 共用这个 posterior（`169-185`）。
- 最终 type weight 只由 `type_query(action_query)` 决定（`186-193`），不看 interval intent、对应 field evidence、teacher uncertainty 或 object identity质量。
- status 是真正的 hidden value；covariance、uncertainty 和完整 future_address 没有在线 P2 consumer。

**问题 `P2-TYPE-ROUTE`（P0）**

- **[源码]** semantic/geometry/status 只有名字和值投影不同，选择空间仍是一套 semantic-owned posterior；geometry 没有自己的 spatial K，status 也能成为廉价独立 value。
- **[日志]** geometry error 明显改善，但 geometry type mass 约 `0.114`，status 约 `0.602`。mass 没有目标值，但源码说明 status 可以绕过未来内容/运动而主导 value，因此这个组合值得担忧。
- **[源码]** contract 规定 visibility/persistence/uncertainty 是 selection/calibration；当前 `status_value` 把 visibility/persistence 直接提升成主要动作 hidden，语义边界过宽。

**[方向]**

- semantic、geometry 使用不同 K/V/posterior：semantic 读 semantic effect；geometry 读 transport、covariance 和 object-to-chart/future-address；
- status 不再与 semantic/geometry 做等价 value 竞争；visibility/persistence/uncertainty 只参与 null/validity/calibration 一次，若确需状态创新则以小型、零中心、单独命名的 innovation 输出；
- type selector 读取 action query、S interval intent 和对应 field evidence，不设 mass quota；
- P2 复用 P1 的 K object posterior，并允许 typed evidence 对同一 object posterior 做 bounded residual，而不是重新从 semantic delta 发明一套 object 地址；
- K/V 的选择权重成对，保持 object/interval/type 轴直到各自消费者。

这对应 MV-WAM 的 manifold-specific route 和 DoT 的 head-specific dock，但保留现有 action flow-matching。

### 4.9 Consequence 与 P3

源码位置：`compiler.py:228-345`

**[源码] 健康部分**

- `protected = factual_base + effect + interaction`，interaction 含 effect 乘法；effect 为零时 interaction 精确为零，consequence 精确回到 P1 fact（`236-250`）。
- precision 和 temporal 都读取 consequence；state-change 是 `0.05`、零值保持的观察变化调制。completion 已被正确移除。

**问题 `P3-DUPLICATE-SEMANTICS`（P1）**

- `protected_base` 已经包含 P1 fact 与 P2 effect；但 P3 又生成 factual lane（来自同一 factual base）和 effect lane（来自同一 effect+interaction）（`293-303,329-335`）。
- **[推断]** 这让 P3 lane gradient 难以解释：effect lane 小可能是 effect 已经通过 protected base 到达，而不是 W 断线；factual lane 也不是新增事实。

**[方向]** 保留一个 protected consequence 主入口；P3 只输出真正新增的 precision、temporal、state-change innovations。若保留 factual/effect 名称，必须改成与 protected base 正交、可证明新增的信息，而不能重复同一代数量。

### 4.10 P3 → bottom：重复 ingress

源码位置：

- `compiler.py:59-76`：`protected_base` 作为 `protected_detail`；
- `clearvla/policy/trunk.py:11921-11973`：typed plan bank；
- `trunk.py:11999-12037`：同一 `protected_base` 再进入 controlled-dynamics context；
- `clearvla/policy/time_domain_mmdit.py:1066-1086,2738-2754`：protected-detail 直接更新 action。

**问题 `BOTTOM-DUPLICATE-INGRESS`（P0）**

- **[源码]** 同一个 `policy_plan_delta_bank.protected_base` 一次作为 bottom `protected_detail` 经专用 reader 直接更新 action；另一次拼到 `context_kv`，先改变 controlled dynamics，再通过 transition memory 进入动作解码。
- 这不是两个不同 owner，而是同一 consequence 的直接和间接双入口；会放大 current/P1 base、混淆 W effect 因果量，并让 P3 typed lane 的解释更困难。

**[方向]**

- 保留已有强探针证据支持的 protected-detail 主入口；
- 从 object capability 的 generic controlled-dynamics `context_kv` 移除同一个 protected base；context 只保留 state、最后的 state/action history；
- P3 innovations 只通过一个显式 role bank；一个语义对象只能有一个 bottom ingress；
- controlled dynamics 仍可产生 rollout transition，但不得再复制 protected consequence。

这直接采用 Faster-WAM SparseMoT/DoT 的“少量明确 exchange point”，不把未来上下文灌入每一层。

### 4.11 Loss ownership

源码位置：`clearvla/experiments/observed_state_lab/policy_runtime_v39.py:3819-4052`

**[源码] 当前做法**

- future 内部权重：successor `0.30`、semantic delta `0.25`、transport `0.15`、covariance `0.05`、visibility `0.08`、persistence `0.07`、uncertainty `0.10`。
- `successor = detached current + semantic_delta`；因此 successor prediction-target 误差与 semantic-delta prediction-target 误差在代数上相同，只是 semantic 项额外增加 normalization/direction（`3922-3950`）。
- structure 内部把 reconstruction、online match、recognizer、coarse action 组合为 `0.25/0.35/0.20/0.20`；其中后三者都继承了 4.4 的平均化 target。
- trunk 在 W2 后还计算一套 raw weighted MSE（`trunk.py:10975-11016`），runtime 又从 prediction/target 重算 canonical loss；这增加了重复实现和语义漂移风险。

**问题 `LOSS-MANIFOLD-DUPLICATION`（P1）**

- 同一 DINO 误差被 successor 和 semantic delta 两次拥有；不同流形又大体套用同一 SmoothL1 模板。
- recognizer/coarse-action 的平均 target 把“好优化的公共方向”加入 structure 压力。

**[方向]**

- successor/delta 合并为一个 canonical future-content objective：保留 absolute anchor、scale-aware magnitude 和 bounded direction，但只记一个 owner；
- transport 使用有界坐标/矩目标，visibility/persistence 使用零中心概率变化，uncertainty 回归 detached teacher dispersion；
- 外部 future/interval 总权重不增加，删除的重复权重不转移；
- canonical loss 只在 runtime 构建一次，trunk 只输出 prediction/target/metrics；
- 不复制 MV-WAM 的 action x0 objective，不改变 action flow matching 主目标。

### 4.12 五步缓存与速度边界

**[源码/契约]** 当前五步 top block count 为 `[1,1,1,0,0,5,0,0]`：G/S/W/P1 一次，P2/P3 随五个 ODE step，Teacher 部署零次。这已经符合 Faster-WAM 的 one-pass future-context 主原则。

**结论**

- 当前主要问题不是“W 每一步重算”；不能为了追论文而重写健康缓存。
- 下一版 dock 必须保持 W K/V 和 P1 factual bank 一次构建；P2 是唯一 future interaction stage，后续 bottom 只做 action refinement。
- 当前 CoarseActionIntent 不依赖 noisy ODE action，因此 W 静态缓存合法；如果未来让 W 读取 noisy action，就会破坏这个边界，必须另立实验而不能静默修改。

## 5. 优先级与问题传播链

| 优先级 | ID | 首个错误边界 | 向下传播 | 先修理由 |
|---|---|---|---|---|
| P0 | `P1-W-OBJECT-BASIS` | global K 没有进入 P1 local read | W effect 与精细 fact 只在聚合 hidden 后相加 | 不先对齐对象，Teacher/W/P2 再好也可能被错位使用 |
| P0 | `P2-TYPE-ROUTE` | semantic-owned posterior 被三类 value 共用 | geometry 被低使用，status 可形成廉价动作 value | 直接决定 W 哪一部分真正进入 action |
| P0 | `S-P1-ALIAS-DUPLICATION` | canonical S 外又注入 mean-goal/last-history | current-only P1 获得多重条件入口 | 可在不删除 S/P1 的前提下消除明显旁路偏置 |
| P0 | `BOTTOM-DUPLICATE-INGRESS` | protected consequence 进入 bottom 两次 | current fact/effect 因果量和 P3 lane 解释混淆 | 单一 dock 的必要条件 |
| P1 | `G-TYPED-IDENTITY` | global binder 只有一套 typed assignment | S/Teacher/W 类型分工变成同一 object 的不同投影 | 决定 object 是否真能支撑 typed dock |
| P1 | `S-AVERAGED-TEACHER` | recognizer 跨时间、跨 K 求均值 | S/W 被训练成公共区间摘要 | 解释平均化为何反复出现 |
| P1 | `TEACHER-TYPED-SMOOTHING` | content-only key + interval uniform mean | W 目标本身偏平滑 | 在 object/P2 接口修好后再处理 |
| P1 | `W-COMMON-MODE` | correlated inputs + shared output heads | W1/W2/四区间变化不足 | 需要结合 S/Teacher 一起修，不能单点加 block |
| P1 | `P3-DUPLICATE-SEMANTICS` | protected base 与 factual/effect lanes 重复 | lane 梯度难解释、信息重复 | 与 bottom 单入口一并处理 |
| P1 | `LOSS-MANIFOLD-DUPLICATION` | 相同 DINO 误差双 owner | 优化预算偏向容易公共方向 | 接线修正后统一 loss |

主传播链：

```text
single typed assignment
  -> recognizer 跨 object/time 平均
  -> S/W common condition
  -> shared W decoder 再平滑
  -> P2 用 semantic posterior 统一选择三类 value
  -> P1/W 不共享 object basis
  -> consequence/P3/bottom 重复入口
  -> W 有 loss/gradient，但 action 收益弱且难归因
```

这条链不是说每个节点都由上一个节点唯一造成；它说明这些问题会同向放大。具体实施必须逐边界消除，不用一个更大的 W 同时掩盖全部问题。

## 6. 下一主线应形成的图

```text
current RGB / DINO / raw / learned flow
  -> local G chart [C,8,8,M]
  -> global K objects
       shared physical assignment
       typed semantic/appearance/geometry verification
       candidate_assignment retained
  -> S observable object intent
       one canonical interval interface
       training-only factorized recognizer target
  -> W1/W2 object-aware future K/V hierarchy
       four real intervals, cached once
       typed semantic / geometry / calibration fields
  -> Object–Chart Future Dock
       P1: K object -> local cell/M/micro current fact read
       P2: same K basis -> one typed future-effect read
  -> zero-preserving consequence
  -> P3 innovation-only compiler
  -> one protected-consequence bottom ingress
  -> unchanged bottom action network
```

核心不是让 W 更大，而是让 W 和 P1 共享 object identity，让不同 effect 类型有不同合法 K/V，并让同一 consequence 只进入 action 一次。

## 7. 实施顺序

1. 定义并画清 `candidate_assignment -> object factual glimpse -> typed future dock` 的真实轴变换；先给现有张量加最小接口，不新增版本化 contract。
2. 去掉 object capability 的 P1 mean-goal/last-history alias，只保留 canonical S interval query；确认 values 仍只来自 RGB/detail。
3. 让 P1 复用 global K `candidate_assignment` 做两级软读，输出 per-object fact 和聚合 fact；不改变图像分辨率、glimpse 数和 micro-grid。
4. 把 P2 分成 semantic/geometry/calibration K/V；status 退出等价 value 竞争；P2 与 P1 共用 K object posterior。
5. 清除 bottom duplicate ingress，并把 P3 收敛为 protected consequence + 真 innovation。
6. 再处理 recognizer factorization、Teacher 有序 K/V、W 轻量 typed heads 和 canonical loss 去重；否则更好的 teacher 仍会被错误 P2 丢弃。
7. 保持五步 cache 和单次 P1 读取，最后才比较 training-only W 与 cached online W；比较使用同数据、seed、batch、八个 epoch、tail/gripper/event 和扰动/OOD 子集。

## 8. 禁止回归

- future teacher、future action/state、recognizer target 永远不进入部署 value；
- 不把 local M hypotheses 叫作 global objects；prior/allocation/existence/validity 不混用；
- object/interval/camera/space/type 轴不得求均值后再 `expand` 伪造；
- 不增加 hard gate、route quota、fixed entropy、forced diversity、forced nonzero flow 或人工梯度；
- 不削弱 P1 的一次高分辨率 current factual read，不恢复第二次 RGB/DINO 读取；
- 不添加无 return 监督的 completion/progress/value；
- 不新增 `_validate_vXXX_*` 或按版本号分支；能力名、小 manifest 和 typed interface 足够；
- 一次 future cache、一个 Object–Chart dock、一个 bottom protected ingress。
