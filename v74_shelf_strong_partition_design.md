# V74 设计稿：货架强分工（B 版）—— 通路语义正式化 + 动态层级读法统一

状态: 设计评审中（依赖 v73 结果与 Phase 0 仪表读数, 部分条款条件触发）
前置: v73 跑完 E3 选出基座; Phase 0 仪表(路由漂移/逐源体积/冗余余弦/读取集中度)
      已随 v73 代码状态上车并有 E1-E3 读数。
原则依据: 货架纪律 —— 世界证据原则(v72) + 名实相符原则(v73), 本轮将其
从"逐案修理"升级为"结构契约"。

---

## 1. B 版原始提案与现状对账

用户提案的 B 版五条, 逐条对照源码现状:

| # | 提案条款 | 现状 | 本轮动作 |
|---|---|---|---|
| b1 | lateral/scan 只给 z/cond | v73 A 版已做(撤架, cond 保留) | 无(继承) |
| b2 | layer_stack 不作静态 workspace 源 | **活路径已成立**: refine 路径 `static_sources.pop("layer")`(policy_v39.py:4519), 全量层记忆由 step 路由的 routed_layer 替代; 仅父类单趟路径(休眠)仍上架 | 正式化: 休眠路径对齐 + 结构断言(§3.2) |
| b3 | 每步只允许 routed_layer/capsule 这类动态层级读法 | routed_layer 已动态(每步重路由); **capsule 是静态的**(decode 前算一次进静态 KV, :4522) | capsule 读法统一, 方向由漂移读数裁决(§3.1) |
| b4 | workspace 保留 trajectory/rollout/transition/progress/routed_layer | v73 后即为此组成(七席: 上述五类 + transition_delta/event 拆分 + capsule) | 无(继承) |
| b5 | action/noisy 只作 query 不作 evidence value | **已成立**: query-only 承诺(:2078 注释), noisy query flag=0(wqscale=0.000 实证), progress 值污染 v72 已切 | 正式化: 白名单断言(§3.3) |

结论: B 版的真实增量 = b3(capsule 读法) + b2/b5 的正式化。改动小, 语义价值大。

## 2. 设计总则

本轮把三条已经事实成立/逐案建立的纪律固化为**结构契约**, 违反即启动失败,
而不是等仪表事后发现:

```
契约一(通路分工): 货架 = 世界证据; cond/z = 任务语义; x_t = 受管制条件通道。
契约二(写入权): 动作与 x_t 对货架只有读权(query), 没有写权(value)。
契约三(层级读法): 全量层记忆不上架; 层级信息只经动态路由(routed_layer)
                或固定容量摘要(capsule)进入货架, 两者读法机制统一。
```

## 3. 具体条款

### 3.1 capsule 读法统一（条件触发, 二选一）

判据: Phase 0 的 `routed_layer` 步间路由漂移 KL(记 drift)。

**方向甲(drift 小, 选择稳定)** —— 统一到"动态但廉价":
capsule 保持静态 token 内容, 但读取权重每步随 route_action 重算
(与 routed_layer 同构), 静态 KV 缓存保留(内容不变, 只有路由动态)。
成本: 每步一次轻量路由计算; 无重复 K/V 投影。
flag: `latent_cvae_capsule_dynamic_route: int = 0`。

**方向乙(drift 大, 逐步重选择被滥用)** —— 统一到"step 0 定选":
routed_layer 与 capsule 的路由权重都在 step 0 由 base_action 一次算定,
后续 refine 步复用 —— "逐样本选择"取代"逐步再选择", 收窄选择型回声灰区。
flag: `latent_cvae_static_selection: int = 0`。

两个方向互斥, 由数据裁决; 设计稿同时给出以避免读数出来后再补设计。
判读: 方向甲看 capsule 份额与 wgeff; 方向乙看 mdwaT 低 t 桶
(选择固定后, 世界证据在低 t 的份额若上升, 说明逐步重选择此前在
为 x_t 依赖让路)与训练稳定性(过渡期 loss 摆动)。

### 3.2 休眠路径正式化（无条件, 行为不变项）

- 父类单趟路径(`_decode_with_z_mmdit` 及其条件组装)与 refine 路径对齐:
  layer 同样不作静态源(pop 或组装时跳过);
- `validate()` 增加断言: `latent_cvae_mmdit_decoder=1` 时若 evidence_sources
  含 `layer` 键即报错 —— 把"活路径靠 pop、休眠路径靠没人走"的脆弱一致性
  变成显式契约。
- 现行配置下计算图零变化(活路径本就 pop), 无需 A/B, 随 v74 代码态直接进。

### 3.3 货架白名单断言（无条件, 行为不变项）

`SemanticEvidenceWorkspace._prepare_sources` 已拒绝未知源名(:2151-2153)。
本轮收紧一层: config 增加 `latent_cvae_workspace_allowed_sources`
(默认 = v73 后的七席名单), 组装侧在 flag 全开状态下装入名单外源名即报错。
作用: 将来任何人(包括我们自己)想"顺手"往货架塞新源, 必须显式改契约,
留下审计痕迹 —— 回应"什么都往上放"的根源问题。

### 3.4 明确不做的事

- 不动 trajectory/rollout/transition/progress 四类源的内容与机制(v73 管辖);
- 不动 cond/z 通路;
- 不做 Slot-attention 竞争读取与 top-k 写入带宽 —— 那是 Phase C 的分区改造,
  立案依赖 Phase 0 的冗余余弦与读取集中度读数, 不与本轮混装;
- progress 跨 chunk 持久化(MemoryVLA-lite)独立立案, 不进本轮。

## 4. 反回声 / 反套利分析

- 方向甲: 路由权重动态化引入的新自由度与 routed_layer 完全同构 ——
  已在灰区清单上, 漂移探针持续监视; 内容(value)不随步变, 无新值流通道。
- 方向乙: 严格收窄自由度, 无新增面。
- §3.2/§3.3 均为断言类, 不改计算。

## 5. 仪表与判读

复用 Phase 0 全套: 路由漂移 KL(裁决 3.1 方向 + 事后验证)、逐源体积、
读取集中度(24-token 画布是否过配)、源间冗余余弦(Phase C 立案素材)。
判决主指标不变: gripper_tail/full rmse; 卫生线: train loss 不超噪声带、
canary 全绿、wgeff 不塌。

## 6. A/B 协议

- 基座: v73 胜者臂。
- v74 = 基座 + 3.1 选定方向的 flag(单变量; 3.2/3.3 为行为不变项随代码态进)。
- 脚本 `scripts/current_v74_shelf_partition.sh`, 链式 wrapper。
- E3 结算。若 3.1 两方向都无立案依据(drift 读数中庸), v74 降级为
  仅 3.2/3.3 的卫生 commit, 不占实验窗口 —— 结构契约的价值不依赖 A/B。

## 7. 实现清单

1. `policy_v39.py`: 3.1 两 flag 的路由改动(方向甲: capsule 路由入 refine 循环;
   方向乙: 路由权重 step0 缓存)、3.2 休眠路径对齐 + validate 断言、
   3.3 白名单 config + 组装侧检查。
2. `train_v40_policy.py`: CLI(3.1 的 flag; 白名单走 config 默认, 不暴露 CLI)。
3. `scripts/current_v74_shelf_partition.sh`。
4. 验证: py_compile + bash -n + flag 两态一致性 grep + 白名单断言的
   故意违规冒烟测试(装一个名单外源名, 确认报错)。

## 8. 风险与回退

- R1 方向选错: 两 flag 都实现, 复跑一次即换方向, 代价一个实验窗口。
- R2 capsule 动态路由的算力开销: 轻量(一次线性路由/步), 但 spb 要盯。
- R3 断言误伤 legacy 配置: 断言仅在 mmdit+workspace 配置下生效,
  legacy 路径不受约束。
- 回退: flag 0 + 断言项本身不改行为, 无迁移成本。
