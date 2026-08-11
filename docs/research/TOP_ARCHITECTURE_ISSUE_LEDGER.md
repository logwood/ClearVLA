# ClearVLA 顶层与主线问题账本

更新：2026-08-11

当前对象：独立 `clearvla/mainline/`，能力名 `object_intent_dynamics_323`。

比较基准：V120 commit `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`；失效迁移对照：V122 commit `ced6f23`；当前候选：HEAD `51f18ad` 上的 schema-20 恢复工作树。

本账本只记录仍能指导实现或实验归因的问题。已经被完整长跑否证的 schema 完成声明、重复版本契约和只证明 shape/非零梯度的条目已删除。当前结构真值见 [`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)。

## 1. 本轮任务边界

本轮不是把旧单体源码原样搬回新包，也不是继续添加版本 validator。目标是：

1. 沿真实数据流逐块比较 V120 与当前独立主线；
2. 保留 V120 中对性能有证据的活动算法，保留新主线中已经更正确的边界与数值实现；
3. 修正当前新增但已由源码和完整日志共同证明有害的机制；
4. 让行为、完整日志和验证结果至少恢复到 V120 的水平，再讨论额外收益。

范围包含当前观测、G/S/W/P1/P2/P3、controlled transition、bottom capacity、训练目标与日志。底层三层 Evidence MMDiT 仍是主路，不重建历史上实际未启用的 variational CVAE posterior 或 hierarchical workspace。

## 2. 受控实验事实

### 2.0 本轮初步结论的地位

刚开始对照时提出的初步结论不是事后附注，而是本轮逐段审查的待证假设：

1. 新主线没有简单“丢掉整个旧 policy”，但抽取时削弱了若干真实活动机制；
2. V122 的失效不只是 RMSE 波动，而是 G 槽公共化、远端动作停滞、夹爪保守化、P3/capacity 梯度消失的联合结构退化；
3. 当前 mainline 比 V120 差，不是因为总参数更少或 CVAE/workspace 没搬回来，而是活动信息路径、目标语义和消费边界发生了漂移；
4. 新主线的 teacher isolation、真实 warp、source-relative flow、typed ownership、静态部署缓存等实现比旧单体更正确，不能为追求早期 loss 相似而退回；
5. 恢复目标必须同时覆盖行为与完整日志；只恢复 aggregate RMSE、shape 或非零梯度不算恢复。

后续源码审查已确认第 1–5 条。它们构成本账本的比较基础；下面每个问题条目记录具体源码证据、修复和仍需长跑验证的部分。

三条正式日志均为同一数据、seed、batch size 8、24-step action horizon、2846 step/epoch、8 epoch：

| 版本 | E1 physical RMSE | best | E8 | 结论 |
| --- | ---: | ---: | ---: | --- |
| V120 | 0.09762 | 0.07931 (E7) | 0.08145 | 当前恢复基线 |
| V122 | 0.09760 | 0.08914 (E6) | 0.09109 | best/final 比 V120 差约 12.4%/11.8% |
| 当前 mainline | 0.10933 | 0.09107 (E7) | 0.09127 | 每轮都差于 V120；best/final 差约 14.8%/12.1% |

### 2.1 V120 的有效状态

- global-K content pair cosine 从 E1 `0.340` 到 E8 `0.508`，没有同质化成单一对象。
- S interval variation 为 `0.068 -> 0.133`，temporal variation 为 `0.024 -> 0.078`，区间信号虽不强但真实存在。
- W2 object pair cosine 从约 `0.508` 到 `0.445`；W2 interval adjacent cosine E8 约 `0.945`。W 不完美，但对象轴仍有内容差异。
- E8 的 `1–4 / 5–12 / 13–24` RMSE 为 `0.02575/0.05959/0.10334`；decoded gripper event F1 为 `0.351`，motion F1 为 `0.830`。
- P3 precision 梯度在 E8 仍约 `1.04e-3`，capacity control/operator 梯度仍非零。
- V120 选中路径是确定性 organizer + 三层 Evidence MMDiT；配置中 `latent_cvae_variational=0`、`latent_cvae_hierarchical_workspace=0`。不能把未运行的历史选项误判成当前缺失算法。
- V120 的日志虽然序列化过一个历史 Stage1 checkpoint 路径，但同一运行身份明确记录 `fresh=1`、`stage1_init=off`、`stage1_initialization_enabled=0`，最终 resolved context 也将 `stage1_checkpoint` 置为 `null`。因此 V120 的优势不是旧 checkpoint 初始化带来的，恢复工作不能依赖古董权重。

### 2.2 V122 暴露的问题

- E8 的 `1–4 / 5–12 / 13–24` RMSE 为 `0.02672/0.06479/0.11644`。近端从 E1 改善约 34%，远端只改善约 4.6%；tail/first 从 `3.31` 升到 `9.24`，是 near 继续拟合而 far 停滞。
- decoded gripper event 预测从 `1424/1357` 退化到 `438/1357`；recall `0.383 -> 0.191`，F1 `0.374 -> 0.289`，说明夹爪越训越保守，而非简单随机波动。
- global-K content cosine `0.625 -> 0.711`，chart overlap `0.381 -> 0.590`，typed posterior L1 仅约 `0.02–0.04`：对象槽越来越公共。
- W 的监督项确实下降，所以“W 没训练”不成立；但 prediction interval variation E8 约 `0.082`，teacher 约 `0.139`，W2 object cosine 约 `0.605`，说明 W 学会的是更公共的默认未来。
- W/coarse-action RMS 增长而 condition interaction 仍约 `0.2`，较大的 action carrier 在旧 `tanh` 合流中会压低 S 的边际灵敏度。
- flow E8 moving/static warp gain 约 `+0.0618/+0.0227`，几何 flow 有价值；global transport prior 却继续减弱。不能删除 flow，应修正它在对象/局部地址中的落点。
- 表征预算从 V120 E8 约 `0.00254` 增到 V122 约 `0.00608`，动作却更差。问题是监督对象和消费路径错误，不是辅助损失太少。
- P3 precision 梯度降到约 `7e-6`，capacity 梯度从 E4 起为零；V122 的“旁路修正”把有用的精细/容量路径一并压断。

### 2.3 当前独立 mainline 的新增证据

- physical RMSE 八轮为 `0.10933, 0.10194, 0.09491, 0.09834, 0.09340, 0.09337, 0.09107, 0.09127`，不是偶发末轮反弹。
- E8 normalized RMSE `0.23622`，first `0.10345`，tail `0.27285`；远端仍是主要残差。
- global-K content pair cosine 在每轮验证都精确为 `1.0`，同时 G reconstruction 从约 `0.938` 降到 `0.759`。这直接证明当前 G loss 可以在完全同质化下继续优化；旧 synthetic 测试没有覆盖这一真实捷径。
- S interval variation `0.0169 -> 0.0418`：比 V122 略有恢复，但远低于 V120，并且输入已经继承同质化 K。
- W2 object pair cosine 同样为 `1.0`；W2 interval adjacent cosine 后期约 `0.970`。W 的对象公共化主要是 G 上游确定性传导，区间公共化还受 S 和 W 写入共同影响。
- E8 W intent/action object interaction RMS 为 `0.234/0.232`，表明两路接线非空，但非空不等于具有对象或区间信息。
- 当前代码的 source-relative flow、真实 warp、RGB photometric anchor、mask target isolation、Teacher no-grad/部署隔离、静态缓存和 typed optimizer ownership 比 V120 更正确，必须保留。
- 逐行核对 V120 已解析配置与 commit 源码后，动作坐标本身不是性能差异来源：V120 正式运行的是 `arm_flow_mode=legacy_independent`、`gripper_field_mode=legacy_handcrafted`、六通道 gripper field、18-D physical chart、独立标准高斯 source 和 `physical_decode_delta_blend=0.25`；当前 codec 的 encode/noise/decode 与该活动分支逐项一致。V120 的 physical flow 也确实按 `0.5*(arm_abs²+arm_delta²)` 与六通道 gripper 均方压回七维尺度，当前 `*_v120_comparable` 使用同一公式和同一 horizon 权重。event/hold row balance 只存在于显式 `*_event_balanced_audit` 审计行，不能再把其较大数值误判为 codec、正式动作目标或基础动作拟合退化。
- 修复前独立主线参数为 `171,355,774 total / 171,253,374 trainable`；V120 为 `235,662,476 / 166,611,570`。总参数减少主要来自删除冻结/跳过 ancestry；修复前 trainable 反而多约 4.64M。性能下降不能用“参数少了”概括。当前工作树的精确参数清单由 architecture manifest 测试锁定。
- 优化器对照确认了一个与日志退化方向一致的实现漂移：V120 对 history proposal 使用 `5e-5 = 0.625×base`，Evidence decoder 常规参数使用 `0.7×base`，operator factor/depth 使用 decoder 的 `2× = 1.4×base` 且显式 no-decay；抽取后的主线却把 proposal、bottom 和 capacity 全部设成 `1.0×base`，capacity basis 还参与 AdamW decay。这会让历史/底部捷径相对 G/S/W 更快，并改变 capacity 的方向学习。修复后 G/S/W/P/transition 保持 `1.0×`，proposal 为 `0.625×`，bottom 为 `0.7×`，capacity basis 为 no-decay `1.4×`；`learning_rate` 始终记录公共 base，三个私有倍率另行记录。

### 2.4 V120 全部批次指标的语义处置

V120 parser 实际得到 `287` 个批次指标；不能把未同名的 `205` 项直接判成缺失，因为其中大部分在新主线中改成了更明确的 owner 名。当前 diagnostic forward/backward 实际产生 `699` 个有限 archival 指标。逐类处置如下：

| V120 类别 | 当前处置 |
| --- | --- |
| action、first/first8/tail、三 horizon band、arm/gripper、decoded/event/motion | 保留并扩展为 physical/normalized、event-balanced/unweighted 及完整验证统计；正式 backward 与 `*_v120_comparable` 都使用 V120 unweighted 物理口径，event-balanced 变体明确标成 detached audit，不能被误认成新的动作几何 |
| flow warp/cycle/smooth/uncertainty/refinement、confidence/occlusion/magnitude | 保留；新增 `-8→-4` 与 `-4→0` 两段以及 feature/RGB、zero-flow 对照 |
| G/S/W/P 和四区间 future normalized error | 保留并按 object/camera/interval/owner 展开，不用旧合并别名作为唯一记录 |
| capacity/effective mass/dwell/operation probability | 以 `bottom_capacity/continue/expected_depth/block update/contraction` 记录；validation 另有 matched-noise no-update/full-update 因果消融 |
| execution value reader/candidate ranking | 确认是唯一重要且非同名替代的活动 V120 机制；具体决策见 `P1-7`，不把昂贵候选图静默搬回 |
| trajectory information weight | V120 正式配置为 `0.0`；旧 score/min/max/effective-fraction 是无效权重诊断。当前保留相同 information-balanced sampler 配置与 run-context summary，不恢复零权重 loss 别名 |
| old workspace/role-compress/typed-refiner/checkpoint-active 等 | 属于被替代接口或未启用 ancestry；只在旧日志 parser 中保留，不污染当前 JSONL |

因此“日志不差于 V120”的验收不是要求复制 287 个旧名字，而是：V120 的活动语义必须有当前、可解释且可因果定位的记录；当前 JSONL 总量及每个 owner 分组下限由执行测试锁定，恢复 gate 再检查八轮行为、结构、梯度和消融。

## 3. 已确认的源码因果链

### P0-1 `G-BINDER-COLLAPSE`

当前 `DenseObjectGrounder` 把同一个 `public_scene_base` 投影后以 `0.5` 加到每个局部候选；所有 K 从一开始就共享强公共方向。主 reconstruction 又以 attached read responsibility 构建 conditional prototype target，assignment 可以和 prototype 一起移动来降低损失。结果是完整数据上 K 全同质，reconstruction 仍下降。

修复决定：

- public scene 只作为受 mask 的全局上下文，不注入每个 binder candidate value；
- 恢复 V120 有证据的 dense spatial mixture reconstruction 为主要当前事实压力；
- conditional prototype 仍保留，但其 responsibility 对 target 构建 detach，阻止 assignment 通过移动 target 逃逸；
- typed consistency 保持近端字段解释压力；不加入 diversity、entropy、slot quota 或硬分配。

验收：真实 smoke 中 K cosine 不再精确为 1；相同事实仍允许相同槽，异质事实的重叠 assignment 必须付出损失。

### P0-2 `P1-GLOBAL-K-ADDRESS-BOTTLENECK`

当前 P1 只围绕 global-K 的单个 camera coordinate 做 3x3 采样，再对 K fact 做中心化。K 一旦相同，坐标、crop 和中心化 key 都相同，精细 RGB/detail 即使被读取也无法形成对象或 query 特异信息。

修复决定：

- P1 的四个 action-basis query 对完整 `[camera,8,8,local-M]` chart 做 query-specific 软读取；
- global-K assignment 只作为先验，不是唯一地址；
- 每个 query/K 得到真实 local posterior、相机条件坐标和一次 3x3 RGB/detail 采样；
- 不重开第二次视觉 bank，不削弱当前高分辨率细节。

验收：P1 chart posterior 和坐标随 query/K 改变；局部 perturbation 只改变对应事实；单次高分辨率读取仍成立。

### P0-3 `BOTTOM-CAPACITY-SEMANTIC-MISIDENTIFICATION`

第一次独立抽取中的 `NestedCapacityOperator` 确实把整个 residual 变成低秩投影，但 schema-20 已恢复的活动 V120 decoder 从未调用它。活动算子是 `NestedLowRankContractionBank`：

```text
output = u - Q diag(1-m(c)) Q^T u
```

因此 full depth 精确 identity；降低 depth 只关闭 `span(Q)` 内的有序方向，正交补始终保留，整体 non-expansive。`c=0` 不是“整个 block 不执行”。此前用非活动 `NestedCapacityOperator` 的 `0 -> zero / 1 -> identity` 单测给了虚假安全感，也让验证中的 `no_updates=capacity_zero` 名称与行为不符。

修复决定：

- 活跃依赖图与测试只以 V120 contraction bank/decoder 为准；旧独立 bottom 不再进入 checkpoint source closure；
- full depth identity、ordered depth 与 non-expansion 由活动 bank 测试；
- 真正的 no-update 验证直接选取所有 host block 之前的 prefix velocity；capacity 只记录 rank retention，不能再承担 block amplitude 语义。

### P1-1 `CAUSAL-VISUAL-HISTORY-OMISSION`

V120 使用 DINO/raw history `(-8,-4,0)` 和两个相邻 flow 区间。当前主线仍保留三行 state history，却只加载 current DINO 和 raw `(-4,0)`，丢掉一帧视觉状态和一段运动变化。S/W 只能从状态/动作历史推断长期变化，更容易退化成 action/history shortcut。

修复决定：恢复三帧 DINO/raw 的 grouped cache read；在线 current fact 取最后一帧，两段 adjacent flow/视觉变化进入 observation/G/S 的合法可观测边界。Teacher 仍单独读取未来 supports，部署不含 future。

### P1-2 `P3-MULTIPLICATIVE-ANNIHILATION`

当前 P3 为杜绝旁路，把 centred P1 detail 乘上 consequence gate，把 S temporal 又乘 consequence/action gate。W 弱或公共时，precision 与 temporal 不是“受后果约束”，而是被一起归零；这与 V122 precision 梯度坍缩和当前性能不恢复一致。

修复决定：

- precision value = query-specific centred local detail + bias-free(detail × consequence interaction)；
- temporal value = S temporal innovation + bias-free(S × consequence interaction)，action 只作有界调制；
- W effect 仍通过 protected consequence 独立进入一次，不恢复 free W residual；
- neutral W 时保留当前精细事实与可观测 temporal，不凭空制造 future effect。

这会明确废止旧“neutral effect 必须让所有 P3 optional lane 都为零”的过强契约；正确零语义是“不产生 W 交互”，不是删除合法当前事实和历史。

### P1-3 `CONTROLLED-TRANSITION-SPATIAL-COLLAPSE`

旧独立抽取只使用 `4 intervals × 4 global-K = 16` 个方向；K 同质后这些方向也高度公共。V120 的活动实现以 `future anchors × cameras × 8×8` 的 dense spatial basis 组织转移。

修复决定：恢复局部空间 basis，在 P1/G 的 dense fact chart 上形成 action-minus-neutral 的 fixed-zero transition；512 个 selector/value 行全部进入 V120 evidence bank，不再先池化成 96 行。只有 event auxiliary context 按四个 horizon milestone 聚合，不能被误认为 transition value 路径。

### P1-4 `HORIZON-WEIGHT-SEMANTIC-DRIFT`

当前 `anchor_horizon_weights` 按 band 等总质量分配，三个 band 的质量约 `31/33/36%`。V120 的实际 per-row 轻微远端强调给三个 band 约 `15/32/53%`。当前实现以“纠正行数别名”为名改变了训练目标，实际上削弱了本来就停滞的远端监督。

修复决定：恢复 V120 的 per-row 线性 band 权重和 first-step protection，24 行均值仍精确为 1；gripper event/hold 平衡保持为独立的行内机制，不再改写 horizon 总质量。

### P1-5 `ACTIVE-LOGGING-BLIND-SPOT`

当前 losses 已计算 action band、arm/gripper/event、flow 等大量指标，但 `ACTIVE_PREFIXES` 缺少裸 `action_`，compact priority 也使用了和 ledger 实际键不一致的名字；主线 audit 工具无法解析 `[mainline-*]`。因此当前日志只显示少数 RMSE/G/S/W 指标，不能达到 V120 的观测能力。

修复决定：

- 恢复 V120 等价的 train/epoch/validation 指标组：physical/native、first/tail、arm/gripper、event/motion、三个 horizon band、flow geometry、G/S/W/P、bottom capacity/controller、owner gradient；
- `metrics.jsonl` 无损保留所有活动指标及精确零值，使 W/P2/flow 真正坍缩与“从未接入”可以区分；nohup 控制台继续隐藏普通零，只保留 owner 梯度、质量守恒和 non-expansive 等决定性契约零值；
- audit 工具支持 mainline JSONL 与 compact lines，不再把 8 epoch 主线解析为 0 rows。
- audit 文本摘要显式投影 schema-20 的 G/S/W/P、teacher、transition、bottom、全部 owner 梯度，以及 normalized/physical/三段 horizon validation；完整 `metric_index` 继续无损保留全部活跃字段。这样“JSONL 里有但人工报告看不到”不再形成第二层观测盲区。
- 正式恢复验收读取运行目录中的 `metrics.jsonl + run_context.json`，严格核对 V120 的 seed/batch/data/split/normalizer 身份、八轮完整性、final 与八轮均值、train-tail、结构/owner 梯度及 matched-noise ablation；缺失证据和实际退化分开报告，但二者都不能宣称恢复。
- V120 的 normalizer 身份是“统计量六位小数后取 12 位 MD5”，新 checkpoint 身份是完整 SHA-256；正式 run context 同时序列化二者。恢复门槛使用 V120-compatible 指纹，checkpoint/resume 继续使用 SHA-256，避免把哈希算法不同误判为数据不同。
- 验证前 `eval_diagnostic_batches` 个 batch 复用同一静态 cache 与同一初始 action noise，分别执行 proposal-zero、bottom no-updates 和 bottom full-updates；记录 physical/normalized RMSE、相对 primary 的 MSE gain、action delta RMSE 与 coverage。它们只用于判断活动路径对动作是正是负，不进入 loss，也不改变正式部署 API。

### P1-6 `V120-OPTIMIZER-GEOMETRY-DRIFT`

源码和完整 launcher 继承链确认，V120 并不是所有活动参数统一学习率：history proposal 为 `5e-5`，G/S/W/P 等新主干为 `8e-5`，Evidence decoder 为主干的 `0.7×`，contraction factor/depth 为 decoder 的 `2×` 且不衰减。独立主线抽取时虽然正确建立了互斥 owner，却把所有 owner 都设成 `8e-5`，并按二维权重的一般规则衰减 capacity basis。该差异会同时提高 history/bottom 相对 G/S/W 的更新速度，并改变低秩方向的 AdamW 动量，不能再当作无关的实现细节。

修复决定：

- 保持 G/S/W/P、observation、controlled transition 的公共 `1.0×base`；
- history proposal 恢复 `0.625×base`；
- bottom query/protected reader/evidence compiler/organizer/MMDiT/execution/heads 使用 `0.7×base`；
- capacity basis 使用 no-decay `1.4×base`；capacity/continue 的控制头仍属于 bottom execution 的 `0.7×base`；
- 公共 `learning_rate` 不再错误取排序后的第一个私有 optimizer group；运行身份保存所有 group 的 LR、weight decay 和参数数，日志单列 proposal/bottom/capacity LR。

这项恢复只对齐 V120 已实际运行的优化压力，不恢复旧 launcher、重叠 owner 或失活模块。它能消除一个明确的对照混杂因素，但动作质量仍必须由同条件八轮长跑验证。

进一步核对 V120 的 active Evidence MMDiT 后确认：启用 execution controller 时，旧 `NestedLowRankContractionBank.depth_weight/depth_bias` 会被显式冻结；日志中的 `grad_evidence_mmdit_operator_capacity` 是整个 contraction module 的梯度，而 `grad_evidence_mmdit_operator_basis` 是其中 basis 的梯度，因此两者在 V120 日志中数值相同。旧 per-operator depth control 不是一条被抽取遗漏的活动能力，不应恢复。当前主线保留唯一 execution capacity head，让它控制 V120 的 full-identity、ordered/non-expansive contraction；basis 仍按 V120 的有效优化尺度训练和记录。

### P1-7 `EXECUTION-VALUE-PARITY-DECISION`

逐项核对 V120 的 287 个真实批次指标后确认：`latent_cvae_mmdit_execution_value_loss_weight=0.05` 在 V120 正式长跑中确实启用，它不是 audit-only execution cost。该 reader 对显式 differentiable candidate chart 的物理动作误差排序进行监督；V120 的 epoch-median value correlation 从 `0.605` 升到 `0.840`，pairwise accuracy 从 `0.85` 升到 `0.94`，top-1 accuracy 从 `0.62` 升到 `0.88`，因此不能把它归类为未使用 ancestry。V122 中该 reader 同样继续学习，但 capacity/operator 梯度从 E4 起归零并且动作性能退化，说明它是健康但不充分的机制，不能单独解释 V120 优势。

schema-20 恢复决定：该 reader 和 differentiable candidate chart 是 V120 活动动作求解器的一部分，不能再以“性能优化”为由用未经实验证明的简化 controller 替代。当前已机械提取 V120 `EvidenceLatentMMDiTActionDecoder`，恢复 candidate value field、prefix candidates 与外部权重 `0.05`；candidate physical prediction 作为 detached target，只训练 value reader。execution cost 仍为 audit-only。日志新增 predicted/target spread、分位数、terminal margin/identity、correlation、pairwise 与 top-1，便于区分“reader 学会排序”和“排序实际改善动作”。

### P0-4 `G-HOST-POLICY-DISCONNECTION`

三层 G role host 确实执行，也确实改写 `public_scene_base`，但修复前 grounder 的在线 competition、candidate value、G3 correction、导出的 K facts 都不消费 hosted public state；它只在 reconstruction query 中出现。于是三个“G 块”可以有梯度和日志，却不改变进入 S/W/P 的对象地址。

修复决定：hosted public context 只进入独立 `public_address_key`，参与 K+null competition 和 G3 correction；candidate update/value 仍只由局部 content/semantic/appearance/geometry/coordinate 构成。这样 G1-G3 真正进入策略地址，同时不会把同一 public value 复制进全部 K。

### P0-5 `P2-INTERVAL-IDENTITY-PRIOR-BYPASS`

P2 原先用 S 的累计 `interval_queries` 作 intent key。该张量含 learned interval identity，即使 goal/history/object innovation 为零也能产生固定时间偏好；这与日志中 P2 窗口分布主要来自固定 temporal prior 一致。

修复决定：P2 只用可观测、数据依赖的 `interval_action_innovations` 排序 W 四区间；W 的 interval 轴仍保留物理时间身份。只替换 learned query identity 不能再改变 P2 effect。

### P0-6 `P3-NOISY-ACTION-TEMPORAL-BYPASS`

P3 temporal base 曾直接乘 noisy ODE action gate。即使 W consequence 为零，temporal lane 也会随 noisy action 改变，从而成为 time-conditioned action adapter，绕开有效 S/W innovation。

修复决定：合法的 S temporal innovation 作为 additive base；noisy action 只能参与 `S × consequence × action` 的 bias-free interaction。neutral W 时 temporal base 保留，但 noisy action 不能从该 lane 制造额外变化。

### P0-7 `TEACHER-RELIABILITY-DOUBLE-DISCOUNT`

原实现只在 null posterior 较大时把 successor 退回 current reference；对“可见但关联熵很高”的样本，它仍会把多个无关 future patch 的空间平均值输出为 target。随后 successor、semantic-delta 和 S recognizer 又各自乘 reliability，导致两种相反错误同时存在：可信度低的含混目标仍不具备中性语义，而真正应学的 current/zero-effect 行又被重复降权。日志中 teacher reliability 偏低、S/W 越来越公共，正符合这一组合缺陷。

修复决定：Teacher-G 先以 association confidence 连续地把高熵 successor 退回 current reference，把 transport/covariance 退回零，并把 future address 退回 unit-mass current address；这些已经中性化的 value/address 与 S recognizer 都使用物理 object validity。reliability 本身仍被校准并作为诊断，但不再二次擦掉 neutral target。这样 reliability 为零时，W 的内容、几何和地址都不能变成 action-owned 自由载体，也不会被迫拟合空间平均 future 或噪声地址。

### P0-8 `CONTROLLED-TRANSITION-NEUTRAL-BIAS`

旧抽取给 real 与 neutral counterfactual 使用不同或带随机性的 coefficient 路径；即使 proposal 为零，两边也可产生不同系数，破坏“动作增量”的零语义。

修复决定：real/neutral 经过同一确定性 coefficient network；neutral 输入为 `zeros_like(proposal.tokens)`；dropout 只作用于相减后的 delta。proposal 为零时 transition value 必须精确为零。

## 4. 保留、恢复和明确不恢复

### 必须保留的新主线实现

- 独立 package/config/train/runtime 与 typed API；
- current-only online API、Teacher no-grad 且每 batch 一次、部署零次；
- source-relative flow 单位、真实 warp、RGB photometric anchor、mask target isolation；
- 18-D physical action codec、三层 Evidence MMDiT、event/motion heads；
- V120 的八个 proposal offsets `(-24,-16,-12,-8,-6,-4,-2,-1)`、4 recent + 3 summary、两层 proposal；
- information-balanced sampler 与 event row weighting 的基本思想；
- checkpoint source closure、fresh/resume 拒绝与 optimizer 单一所有权。

### 从 V120 恢复或等价重建

- 三帧 causal visual history 与两段 adjacent motion；
- 对真实 local chart 的 P1 多 glimpse 地址读取；
- dense spatial controlled-transition basis；
- V120 horizon per-row 权重；
- 完整指标和验证口径。

### 不恢复

- 额外于 V120 活动 decoder 的第二套 execution/capacity 重写；
- 自由 uncertainty NLL；
- 未监督 public W residual；
- 实际未启用的 variational CVAE posterior / hierarchical workspace；
- scalar progress、phase label、completion terminal、hard gate、route quota、固定 entropy、forced slot diversity、forced nonzero flow 或人工梯度。

## 5. 实施顺序与状态

| 顺序 | 项目 | 状态 |
| --- | --- | --- |
| 0 | V120/V122/current 全 epoch、源码与初步结论对照 | 已完成；初步结论已转成待证假设并逐项确认 |
| 1 | `BOTTOM-CAPACITY-SEMANTIC-MISIDENTIFICATION` | 已修正；活动 V120 bank 的 full identity/ordered non-expansion 测试通过，true no-update 改用 prefix boundary |
| 2 | `G-BINDER-COLLAPSE` + `G-HOST-POLICY-DISCONNECTION` | 已实现；真实长跑效果待验 |
| 3 | `P1-GLOBAL-K-ADDRESS-BOTTLENECK` | 已实现；full local chart、普通 autograd 与单次 packed read 单测通过 |
| 4 | causal three-frame visual history + dense transition | 已实现；两段 flow 与完整 512-row selector/value transition 单测通过 |
| 5 | additive typed P3 + P2/P3 bypass 修复 + V120 horizon weights | 已实现；neutral/identity 与依赖单测通过 |
| 6 | teacher 高熵 fallback、reliability 双折扣与 neutral-transition 修复 | 已实现；目标/零语义单测通过 |
| 7 | full active logging + mainline audit parser | 已实现；活动指标包含 exact weighted contributions、独立闭合残差、V120 正式 action loss、event-balanced audit、execution-value 排序/幅度/terminal 诊断，并以总量及 G/S/W/P/transition/bottom/owner-gradient 分组下限锁定；无损 JSONL、compact semantic rows、true no-update/full-update 消融与跨版本 parser 均通过本地验证，仍需真实新日志验收 |
| 8 | V120 optimizer geometry 恢复 | 已实现；proposal/bottom/capacity 比率、capacity no-decay、base-LR 日志语义均有单测 |
| 9 | provenance / numerics-autograd / runtime 三轮审查 | provenance、数值/普通 autograd、五步静态缓存与评测干预隔离已完成；活动 graph 不再经旧 observation/bottom 原型，rollout 只保留 selector 而不能绕过 centered transition value；零初始化的首步启动边界经过第二个真实 optimizer step 后，全部 trainable tensor 均获得非零普通梯度；`132` 项回归通过，scoped Pyright `0 errors`；生产显存仍待服务器 smoke |
| 10 | BF16 smoke 与服务器 V120 受控八轮比较 | CPU BF16 完整前后向已通过；本机无 CUDA，服务器 batch-1/batch-8 显存与八轮结果待执行，这是性能恢复声明的必要条件 |
| 11 | V120 execution-value 语义对照 | 已确认其为活动且可学习但不充分的机制；schema-20 已恢复原 candidate chart/value reader 与 `0.05` 监督，execution cost 保持 audit-only，长跑需同时核对 ranking 与 action utility |

## 6. 验收口径

静态与单元验收：

- capacity full depth 为 identity、降低 depth 只关闭有序 Q 子空间且 non-expansive；true no-update 输出等于 block 前 prefix，不能用 capacity=0 冒充；
- public scene perturbation不能以同一 additive value复制到所有 G candidates；
- object permutation 与 local-chart permutation 在 G→P1→W/P2 全程等变；
- P1 query/K 能选择不同空间支持，且视觉 bank 只读取一次；
- future support 替换只改变 target/loss，部署 action 不变；
- neutral W 不制造 W interaction，但不删除当前 P1 detail 与合法 S temporal；
- controlled transition 对 proposal=neutral exact zero，并保留空间差异；
- 五步部署不重建 observation/G/S/W/P1/Teacher；所有 trainable 参数有且仅有一个 optimizer owner；允许零初始化边界在首步令其上游为零，但第二个真实 optimizer step 后不得仍有 trainable tensor 保持 exact-zero 梯度。

长跑验收：

- 先与 V120 同数据/seed/batch 比较全部 8 epoch，不只看首轮或 best；
- physical RMSE、normalized RMSE、first/tail、arm/gripper、event/motion、三 horizon bands 均不得差于 V120；
- G object cosine 不能再精确为 1，且 reconstruction 下降不能以 K 公共化换取；
- S/W prediction variation 应接近 teacher，而不是只增加 RMS；
- P3 precision 与 capacity 梯度不能再次在后期消失；
- flow 的 moving/static warp gain不得因地址修复而丢失；
- 若边界和所有权均正确而 action 仍无增益，再归类为数据可识别性问题，不继续堆契约或辅助 loss。
