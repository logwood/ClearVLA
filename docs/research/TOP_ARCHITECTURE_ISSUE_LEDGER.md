# ClearVLA 顶层问题账本

更新：2026-08-08

对象：当前 `object_intent_dynamics_323` schema 3 工作树（默认实验标签 v121）

范围：Pre-G 之后的 G / S / Teacher / W / P1 / P2 / P3、相邻 loss、缓存和 bottom ingress

不在范围：Evidence MMDiT、CVAE、workspace、execution controller 内部重做

当前执行图和禁止输入以
[`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)
为准。本账本只保留两类内容：已经由源码关闭的结构问题，以及仍须由新实验判断的问题。旧 V120 行号、旧五路 P3 和已经失效的候选修补不再保留。

## 1. 历史证据的有效边界

三个完整 V120 仍然说明：

- 最好 action RMSE 约 `0.07931`，但后期会轻微反弹；tail、gripper 和 event 仍是难点。
- W 有 loss 和梯度，不是完全断线；但 prediction interval variation 约 `0.0965`，低于 teacher 约 `0.140`，且 prediction adjacent cosine 约 `0.915`，高于 teacher 约 `0.856`。W 明显比目标更公共、更平滑。
- transport normalized error 曾改善到约 `0.61–0.65`，但旧 P2 的 geometry value mass 仍很低、status value 反而占主导。这证明“几何 loss 下降”不等于“几何被 policy 使用”。
- P1 detail/spatial 指标随 RMSE 改善而增强，因此不能削弱 P1 来强迫 W 显得重要。
- completion 曾塌成全局偏置并直接影响 terminal/execution。它不可识别，已经移除；其日志不能再解释成阶段学习。
- `grab_pen_single` 只有单一语言条件。任何语言区分能力、长期 goal 泛化能力都不能由该数据单独证明。

这些数据属于 schema-2 祖先，只用于说明为什么要改结构；不能证明 schema 3 已经有效。

## 2. 已由当前源码关闭的问题

“关闭”只表示结构缺陷不再存在，不等于已经获得动作收益。

| 旧问题 | 当前关闭方式 | 可执行边界 |
| --- | --- | --- |
| `P1-W-OBJECT-BASIS`：P1 在 local lattice、W 在 global K，二者只在聚合 hidden 相遇 | P1 复用同一个 `candidate_assignment`，输出 `ObjectFactualDock.fact_by_object/object_posterior/chart_posterior/coordinates`；P2 在同一 K 基底读 W | object permutation 必须贯穿 G→S→Teacher→P1/P2 等变 |
| `G-TYPED-IDENTITY`：semantic/appearance/geometry 只是同一内容的改名投影 | 只有一个物理 K+null assignment；三类 typed verifier 只能在其 support 内作 bounded correction，零初始化时精确继承物理 posterior | typed posterior 不能制造新 K、null 或无效 support |
| `G-COORDINATE-RECON-BYPASS`：共享坐标解码器可独自降低 DINO 重建而 K 槽继续同质 | 原有重建总预算内，75% 由不含坐标项的 object-prototype 重建承担，25% 才允许坐标细化 | 不增加 diversity/entropy 目标；分别记录 prototype 与 spatial-refinement MSE |
| `S-P1-ALIAS-DUPLICATION`：canonical S 外又注入 mean-goal/last-history | object 主线 P1 只接 canonical interval query；goal/history/object/action/state 保持 factorized K/V | P1 value 仍只来自当前高分辨率 observation |
| `S-AVERAGED-TEACHER`：recognizer 跨时间和 K 求平均 | recognizer 分别输出 interval action/state/object-key/object-value target | teacher target 只训练 S，不成为部署 value |
| `TEACHER-COMMON-TIME`：所有对象共享同一 support 时间权重 | temporal reliability 现在为 `[B,F,K]`，每个对象独立选择 support；stable successor 与 end-biased semantic change 分开 | 替换 future support 只能改变 target/loss |
| `W-COMMON-MODE`：重复当前 base、W1/W2 共用输出头 | 当前对象只进入 W 一次；S object K/V 是 future-oriented innovation；near/far heads 参数互斥；W2 保留两枚有序 W1 token | W 仍只有两块，不靠增大网络掩盖接口错误 |
| `P2-TYPE-ROUTE`：semantic posterior 统治 geometry/status 三类 value | semantic 与 geometry 拥有独立 query/key/posterior/value；future address 参与几何坐标；status 只校准 null/validity | semantic-only perturbation 不得改变 geometry posterior |
| `P3-DUPLICATE-SEMANTICS`：protected base 外又复制 factual/effect lane | P3 只保留 precision、temporal、state-change 三个 innovation | protected consequence 位于 optional route softmax 外 |
| `BOTTOM-DUPLICATE-INGRESS`：同一 consequence 同时走 protected detail、controlled dynamics 或旧 layer contract | P2/P3 与 controlled dynamics 都用 pre-top trajectory seed；consequence 只经 typed P3 bank；object capability 不再构建/执行 midcut 与 layer-contract tower | bottom 不再从 P 修改后的 trajectory 或 post-P layer readout 间接重读 consequence；Evidence adapter 的 layer 输入是显式零行 |
| `LOSS-MANIFOLD-DUPLICATION`：trunk/runtime 双记 loss，successor 与 delta 代数相同 | runtime 是唯一 canonical loss owner；Teacher 的 stable successor 与 ordered semantic end target 不同 | 删除的 loss 权重不转移到新项 |
| `COMPLETION-BIAS`：无监督 completion 成为 terminal 全局偏置 | completion/terminal lane 退出 object 主线；仅保留零中心 observable state-change 小调制 | frame progress 只可 audit，不可进入 forward/loss |

## 3. 论文机制在当前实现中的边界

- **Fast-WAM** 的可迁移部分是训练期未来监督与部署值隔离；当前 Teacher/recognizer 只塑造在线分支，不进入部署缓存。不能据此恢复像素视频生成或把 future token 塞给 action。
- **Faster-WAM（future conditioning）** 的可迁移部分是一次建立、少量显式交互和 K/V 成对缓存；当前 G/S/W/P1 一次缓存，P2 是唯一 future dock。不能把 W 注入每层，也不能把论文的深度 interval 混同于这里的时间区间。
- **Faster-WAM（Dock of Transformer）** 的可迁移部分是先对齐 world/action 的身份与坐标基底；`ObjectFactualDock` 正是 global K、local chart 和 action query 的显式交点。当前没有对应 3D/1D RoPE 证据，因此不生造 RoPE 修复。
- **MV-WAM** 的可迁移部分是异质信息使用不同 K/V、目标和单向信息流；因此 semantic/geometry/status 被拆开，Teacher→W→P 单向。没有 return、失败或 rollback 数据时，不引入 value/completion/progress。

共同结论不是“W 越大越好”，而是：future teacher 不泄漏、交换点要少而明确、身份/坐标先对齐、异质信息不进同一个汤。

## 4. 当前真实主路

```text
local G hypotheses [B,C,8,8,M,*]
  -> one physical K+null global assignment
ObjectFactSet [B,K,*], K=4
  -> factorized S [B,4,K,H] + [B,4,H] + [B,24,H]
  -> W1/W2 FutureObjectDynamics [B,4,K,*]

ObjectFactSet.candidate_assignment
  -> one P1 high-resolution micro-read
  -> ObjectFactualDock [B,T,Q,K,H] + K+null/chart posterior

ObjectFactualDock + typed FutureObjectDynamics
  -> semantic/geometry P2 selectors
  -> one zero-preserving consequence
  -> P3 precision/temporal/state-change innovations
  -> one protected bottom ingress
```

P1 的成熟 aggregate fact 被保留；per-object dock 是与 W 对齐的精细事实载体，不是用较弱的 K 重建替换 P1。W 的目的也不是取代 P1，而是在同一对象/坐标基底上提供 P1 当前事实无法包含的未来后果。

## 5. 仍需新实验判断的问题

### E1 — global K 是否真的形成可用对象

关注：

- `object_pair_cos` 与 `chart_pair_overlap` 是否持续接近 1；
- semantic/appearance/geometry posterior L1 是否始终为 0；
- object existence、physical validity、allocation share、null mass 是否各自保持合理且不互相伪装；
- 不同相机、gripper event 与高运动样本中的 K 是否仍稳定。

没有固定 diversity/entropy 目标。若内容相似但 chart 不同，可能是合法的同类对象；只有内容和 chart 都同质、且干预也不可分，才能判为 object collapse。

### E2 — Teacher 是否提供足够且正确的未来差异

关注四区间、对象、相机和空间分轴的：semantic/appearance margin、null、visibility、uncertainty、target variation、adjacent cosine、transport variation。

若 target 本身近乎公共，优先归因数据运动/未来 DINO 可识别性，不再通过强迫 W diversity 伪造差异。若 target 有差异而 prediction 没有，才归因 W 学习或 loss 尺度。

### E3 — W1/W2 是否从“有梯度”变成“有分工”

关注：

- near/far interval normalized error 的完整 epoch 曲线；
- prediction variation 与 target variation 的比例；
- W1/W2 object pair cosine、interval adjacent cosine；
- goal/action/state/object/typed innovation 的相对幅度；
- W zero/shuffle 是否先改变 P2/consequence，再改变 action。

W1/W2 仍相似时，先看 Teacher target 和 S factorized target 是否也相似；不能仅凭名称或梯度强行判为 W head 错误。

### E4 — P2 是否真正使用 semantic 与 geometry

关注两套独立 posterior 的 score、entropy、max、null、四区间 mass，以及最终 semantic/geometry type mass。mass 没有人工目标；健康含义是它随样本和干预变化，而不是必须 50/50。

必须分别做 semantic zero/shuffle、transport/address zero/shuffle。joint intervention 只能补充，不能掩盖是哪条路径有效。若 representation 边界变化而 consequence/action 不变，说明仍被下游补偿或忽略。

### E5 — P1 强而 W 弱是否仍存在

P1 强不是错误。验收目标是：保持 P1 精细当前事实，同时 W 在 precision-critical、tail、gripper/event 与较长 horizon 子集提供互补增益。禁止削弱 raw/detail、减少 glimpse 或增加硬门来制造 W 使用率。

### E6 — S 的可识别性与长程能力

单任务单语言只能检验历史/对象变化和四区间组织，不能证明语言泛化。frame progress 与 S 指标只作 audit。若 factorized boundary 正确但 S 仍固定，下一研究项应是更丰富目标/轨迹数据或 training-only plan recognizer，而不是恢复 scalar progress/伪阶段标签。

### E7 — 泛化平台、速度和显存

至少比较三个验证点和全部八个 epoch：full/first/tail、arm/gripper、event、四 horizon、precision-critical 子集，同时看 train/val gap。结构因果正确但动作无增益时，归入数据/目标可识别性，不继续无止境改接线。

性能门：本地微型 BF16 forward/backward 小于 8 GiB；生产 batch 8 总进程不超过 22 GiB；Teacher 每训练 batch 一次、部署零次；五步采样不得重复构建 G/S/W/P1。

## 6. 新日志的最低信息集

- G：reconstruction、existence、validity、allocation、null、mass error、object content cosine、chart overlap、三组 typed-posterior L1。
- S：goal/history/object/type attention；action/state/object K/V interval variation；各 innovation RMS；observable state-change。
- Teacher/W：四区间逐项 normalized error；target/pred variation 与 cosine；per-object reliability/null/margin/transport。
- P2：semantic/geometry 各自的 score/logit/posterior/null/interval mass、type mass、calibration。
- P3/bottom：consequence ratio、三条 innovation、各 owner gradient、global preclip。
- 训练/验证：所有 epoch 的 action、first/tail、arm/gripper/event 与 horizon 指标。

旧 `p3_factual`、`p3_effect`、status value、completion、错误 argmax 名称和 inactive ancestry 零梯度不属于 v121 活跃语义。

## 7. 禁止回归

- future teacher、future action/state、recognizer target 永远不进入部署 value。
- 不把 local M 叫 global object；prior/allocation/existence/validity 不混用。
- 不通过 reduce-then-expand 伪造 object/interval/camera/type 轴。
- 不增加 hard gate、quota、固定 entropy、forced diversity、forced flow 或人工梯度。
- 不削弱 P1 的一次高分辨率当前事实读取，不恢复第二次 RGB/DINO 读取。
- 不增加无 return 监督的 completion/progress/value。
- 不新增 `_validate_vXXX_*` 或按版本号分支；能力名、小 manifest、typed interface 和真正的边界测试足够。
- 保持一个 future cache、一个 ObjectFactualDock、一个 protected consequence bottom ingress。
