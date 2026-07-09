# V74 设计稿（重写）：货架分区 —— B 版强分工真身

状态: 设计评审中
前身说明: 本稿替换上一版"契约固化"稿。上一版把强分工写弱了 ——
  内容正名(v73)与写入契约解决的是"汤里放什么", 本稿解决"分锅":
  七个源不再进同一个 softmax、不再被同一组无差别 query 混读。
前置: v73 跑完 E3 选出基座。Phase 0 仪表不再是本轮的立案条件
  (分不分锅已由设计决定), 只裁决细节(§3.4)。

---

## 1. 问题（为什么单一市场是结构病而不只是卫生病）

现状: SemanticEvidenceWorkspace 用 24 个无类型 query token 对全部源的
token 做一次 cross-attention(2 block), 输出 24 个混合 token 作为单一
"workspace"组进 MMDiT。三个后果:

1. **类型间伪竞争**: 事件证据和几何证据争同一份注意力, 但它们本不该
   竞争 —— 动作在低 t 需要几何、在事件窗需要时机, 是"都要", 不是"二选一"。
   单一 softmax 把互补关系强制成了竞争关系。
2. **语义靠数据重新发现**: 63 条 demo 下, "transition 族管时机、trajectory
   管几何"这种我们先验就知道的分工, 要靠梯度从零学 —— 学费是甩尾期
   (wroute 冲 0.42 再回落)和永久印记。
3. **审计粒度不足**: mdwa 是一个总数, 世界证据内部谁在承重、在哪个 t 段
   承重, 读不出来 —— S3 问题拖了这么久, 根子就是这个读数太粗。

成熟解法谱系(MemoryVLA 的两层类型化条件、RIMs 的通道承诺、GWT 的
有限带宽)都指向同一件事: **类型先验写进接线**。

## 2. 分区方案

### 2.1 通道划分（v73 后七席 → 四通道）

```
CH-geom  几何/动力 : trajectory(24), rollout(N)      —— 手往哪走、世界怎么变
CH-event 事件/时机 : transition_delta(1), transition_event(24, v73 真证据)
CH-state 任务状态  : progress(P), capsule(C)          —— 进行到哪、场景摘要
CH-layer 层级语义  : routed_layer(24)                 —— 主干深层的路由读出
```

划分是先验知识, 不由数据裁决; Phase 0 读数只影响细节归属(§3.4)。

### 2.2 读取机制: 通道内独立读, 通道间在 MMDiT 处计量融合

**workspace 侧**(每通道一次受限读):
- block 参数共享(两个 SemanticEvidenceWorkspaceBlock 不变, 无参数爆炸),
  每通道独立调用: 该通道的 query 集 × 仅该通道的源 token 记忆;
- 通道 query: CH-geom/CH-event/CH-layer 用 24 个 horizon 对齐 query
  (时序结构是这三类的本体); CH-state 用小容量 query(4 槽, 任务状态
  不需要 24 个时间位);
- 通道各自保留 -log(count) 源先验、静态 KV 缓存(按通道切分)、
  causal self-attn(时序通道)；
- 每通道加一个通道 type embed + 轻量输出 LN(共享主参数, 通道特异只有
  embed 级参数)。

**MMDiT 侧**(通道间分配在这里发生, 全程计量):
- cond 组从 [workspace(24), noisy(24)] 变为
  [geom(24), event(24), state(4), layer(24), noisy(24)];
- **组先验推广**(原 S2 修复在此从可选变为必需): 每组 key 加
  −log(group_len/action_len) 的 logit 偏置, 组间竞争先验公平;
- 逐组注意力份额 + 逐组 t 分层(mdnaT 机制直接推广): 
  mdwa 拆成 mdwa_geo / mdwa_evt / mdwa_sta / mdwa_lay,
  每个都有 t0/t1/t2 分层 —— S3 从"世界证据总量"细化到
  "哪类世界证据在哪个 t 段承重"。

**关键性质**: 类型间不再争 workspace 内的同一 softmax; 争用只发生在
MMDiT 的动作读取处 —— 那里每一份分配都有份额仪表、组先验公平、
且受 t 分层监视。混合从"暗处的一锅"变成"明处的计量配电盘"。

### 2.3 计算与参数量

- workspace QK FLOPs: 每个记忆 token 仍只被~24 个 query 读一次
  (原来是 24 query × 全部 token, 现在是分通道各自 24×通道内 token),
  总量基本持平; state 通道 query 从 24 降到 4, 略省。
- MMDiT cond 从 48 token 变 100 token: 动作侧 attention 成本上升约 2 倍
  (48→100 key), 绝对量仍小(h=hidden, 24 action query)。spb 预算 +5% 内。
- 新参数: 4 个通道 embed + state 通道 query(4×h) + 输出 LN ×4,
  合计 < 0.1M。共享 block 不变, checkpoint 大部分兼容
  (workspace query 24→按通道重排, 需要重新初始化的只有 query/embed)。

### 2.4 兼容与 flag

`latent_cvae_workspace_partition: int = 0`(config + validate + CLI)。
- 0 = v73 行为原样(单一市场), 代码路径完整保留;
- 1 = 四通道分区。一个 flag 一个变量: 分区是一次连贯重组, 不拆散测。
- 上一稿的两个行为不变项随本稿代码态直接进(不占变量):
  休眠路径 layer 对齐 + validate 断言; 货架白名单改为**按通道**白名单
  (新源必须声明通道, 治"什么都往上放"的根)。

## 3. 判读与细节裁决

### 3.1 判决主指标
gripper_tail/full rmse(不动的老规矩); 卫生线: train loss 噪声带内、
canary 全绿、spb +5% 内。

### 3.2 分区专属预期
- 甩尾期缩短: E1 前 600 batch 的份额轨迹应明显比 v69/v72 的
  wroute/mdna 过冲平缓(类型分工不再需要从零发现);
- mdwa_evt 在事件窗(结合 gfnehr 的 event mask 逻辑)应显著高于 hold 段;
- mdwa_geo 低 t 桶应接过 v72 读数里 workspace 低 t 份额的主力
  (0.139 那一份的归属第一次可读);
- 若各通道份额与单一市场时代的内部份额几乎相同、val 也不动 ——
  说明市场早已自发分工, 分区只买到审计粒度; 这也是有效结果, 不算失败。

### 3.3 风险信号
- 某通道份额长期≈组先验(市场不读它): 通道内容或必要性存疑,
  比单一市场时代更容易定位;
- state 通道 4 槽若饱和(读取集中度打满), 扩容;
- MMDiT 组先验推广后 mdna 绝对值会平移(先验变了), 与 v72/v73 对比
  需重新基线 —— 判读手册里写死, 防止又一次"被几个数字带偏"。

### 3.4 Phase 0 读数裁决的细节(不影响开工)
- capsule 归 CH-state 还是 CH-layer: 看它与 routed_layer 的冗余余弦;
- routed_layer 步间漂移大: 该通道内改 step-0 定选(上一稿方向乙,
  收窄为通道内细节);
- 通道内是否需要竞争读取(slot-attention 式 explaining-away):
  看通道内源间冗余余弦, 留作 v75+。

## 4. A/B 协议

- 基座: v73 胜者臂; v74 = 基座 + `--latent-cvae-workspace-partition 1`。
- 脚本 `scripts/current_v74_shelf_partition.sh`, 链式 wrapper,
  头注释含 §3 判读手册全文。
- E1 看甩尾形态与 canary, E3 结算。

## 5. 实现清单

1. `policy_v39.py`:
   - config/validate/CLI: partition flag + 通道白名单;
   - `SemanticEvidenceWorkspace`: 增加分通道 forward 路径(共享 block,
     per-channel query/embed/LN; 静态 KV 按通道 prepare);
   - `_mmdit_condition_tokens`: 五组 cond 组装 + 组先验偏置推广
     (改造 `_action_key_bias` 为多组版本);
   - `LatentCVAEMMDiTBlock`: 组份额 metrics 从 2 组推广到 5 组
     (per-sample rows 机制沿用 v72);
   - 两个行为不变项(休眠路径对齐、白名单断言)。
2. `policy_runtime_v36_3/v39.py`: 新组份额/分层键转发 + console
   (mdwa 拆四 + 各自 T 分层; 旧 mdwa 保留为四通道之和, 兼容跨版本对比)。
3. `train_v40_policy.py` + 脚本。
4. 验证: py_compile、bash -n、flag 两态一致性 grep、
   白名单故意违规冒烟、组份额之和≈旧 mdwa 的数值一致性冒烟。

## 6. 风险与回退

- R1 检查点迁移: workspace query 重排导致部分参数重初始化 ——
  v74 从头训(8 epoch 全程), 不做热迁移, 避免混淆归因;
- R2 组先验推广改变 mdna 语义: §3.3 已写入判读手册;
- R3 分区后某通道退化: flag 0 一键回单一市场; 通道级白名单保留
  (它不依赖分区)。
- R4 与 Phase C 的关系: 本稿就是 Phase C 的第一阶段(类型承诺);
  竞争读取/top-k 带宽是其后续阶段, 依据 §3.4 读数另行立案。
