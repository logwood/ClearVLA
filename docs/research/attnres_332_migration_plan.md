# ClearVLA 3-3-2 AttnRes 结构迁移计划

状态：未来设计文档，不修改 V102 的当前训练实现。

基线：V102 `anchor_world_late_raw_detail` 单阶段端到端结构。

目标：在不破坏空间地址、角色所有权、时间所有权和自然梯度的前提下，将 AttnRes 用于信息组织、筛选和跨深度检索，并为以后较完整的结构替换保留清晰边界。

空间地址的详细设计以
[Soft Multi-Resolution Address Lattice](soft_multires_address_lattice.md)
为准：8×8 cell 是带多个软地址 slot 的自主查询单元，不是与高分辨率
patch 一一对应的硬指针。本文中的 grounding/world AttnRes 只负责组织
depth/role delta，并生成地址 query；高分辨率 coarse/fine posterior 由该
地址格形成。

## 1. 结论先行

AttnRes 适合 ClearVLA，但不适合无约束地替换全部残差连接。

当前最合适的四个位置是：

1. `grounding(3) -> world(3)`：从三个 grounding 增量中选择形成 world 的依据；
2. `world(3) -> policy(2)`：按动作 horizon/basis 选择世界增量，同时接收受保护的 late raw detail；
3. 顶层 policy workspace 到底层 Evidence MMDiT：替换当前固定的单一 workspace 融合口径；
4. `EvidenceConditionOrganizer`：以角色化深度检索替换对少量 layer rows 的单向 GRU 扫描。

以后若上述边界验证成功，可以进一步将每个角色组内部的普通累计残差改成 role-local Delta AttnRes。第一版不应改 flow/DINO 的空间匹配、late-detail 的相机内空间注意力、执行控制器或所有 FFN 残差。

## 2. 依据与适用范围

原始 AttnRes 用输入相关的深度注意力替换固定单位权重的历史层累加；Block AttnRes 则只在块边界选择历史块表示，以降低全深度路由开销。官方说明和伪代码见：

- [Attention Residuals 论文](https://arxiv.org/abs/2603.15031)
- [MoonshotAI 官方实现与 Block AttnRes 伪代码](https://github.com/MoonshotAI/Attention-Residuals)

ClearVLA 不应直接复制 LLM 版本，原因是当前网络不是同质 Transformer 深堆叠：

- 顶层八块具有 `grounding/world/policy = 3/3/2` 的不同写权限；
- world 在 V102 中只能写 `[anchor, camera]`，不能写 xy-specific residual；
- raw detail 保留 `[camera, xy]` 空间结构，并在 world→policy 边界读取；
- 底层 MMDiT 同时承担 action self update、typed evidence read 和执行控制；
- 全局 `z` 必须保持 clean-intent 语义，不能被 noisy action 反向污染。

因此，本计划采用以下两个后续工作的启发：

- [Delta Attention Residuals](https://arxiv.org/abs/2605.18855)：优先路由每层产生的 delta，而不是高度重复的累计 hidden state；
- [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694)：路由 key 使用低维投影，value 保持完整 hidden dimension；
- [Attention Sinks and Outliers in Attention Residuals](https://arxiv.org/abs/2605.17887)：必须记录 sink、激活异常和 null/current-route 行为，不能假定 softmax 路由天然稳定。

这些论文主要在语言模型上验证；迁移到 ClearVLA 的收益仍需通过动作级干预和验证集证明。

## 3. V102 当前边界

### 3.1 顶层 3-3-2

`clearvla/policy/trunk.py` 根据配置构造三个 grounding、三个 world 和两个 policy block。每个 `TemporalDynamicsBoundDiTBlock` 的写权限为：

- grounding：可写 clean context、stage、rollout 等组织区域，不写 trajectory；
- world：只写 stage/rollout；
- policy：只写 trajectory。

policy 可以通过定向 canvas attention 读取 world rollout，但不能重新读取原始视觉 memory。这个权限边界必须保留。

### 3.2 V102 world 所有权

`clearvla/policy/trunk_primitives.py::_structure_world_rollout_update()` 将每次 world update 池化成每个 `[anchor, camera]` 一个向量，然后广播回 xy slots。world 可以组织时间和相机信息，但不能制造局部空间细节。

这不是待删除的瓶颈，而是一个显式角色契约。AttnRes 必须在相同 `[anchor, camera]` 语义下路由 world delta。

### 3.3 V102 late raw detail

`clearvla/policy/flow_dino_evidence.py` 编译 observation-only 的高频 residual bank：

- selector 保留 source-chart 的 camera、xy 和 raw-type 身份；
- value 是 post-reader high-frequency residual；
- noisy action 和 world token 不进入 detail bank 的编译；
- detail bank 不再提前融合进 DINO/world memory。

`LateRawDetailPolicyReader` 在 world→policy 边界，以 `[time, basis, camera]` query 在每个相机自己的 xy chart 内做空间读取，并用固定 scale 加到 trajectory。

这一步解决“从图像哪里读”，不是“从哪些深度和角色读”。它本身不应被 AttnRes 替换。

### 3.4 顶层到底层

顶层最终导出：

```text
policy_workspace = final_trajectory - normalized_trajectory_seed
```

底层 Evidence MMDiT 先构造 noisy action state、semantic seed 和 horizon position，再将 horizon-pooled policy workspace 通过固定方差融合写入 action stream。

严格角色模式只把最后两个 policy layer contracts 交给最终 Evidence decoder；当前 organizer 的 layer scan 因而只看到两个 terminal policy rows，而不是完整的 3-3-2 角色过程。

## 4. 两种迁移模式

### 4.1 Bridge 模式：第一阶段

保留当前主路径，只在角色边界增加一个受约束的 Delta AttnRes 检索：

```text
x_boundary = x_current + fixed_scale * AttnRes(allowed_deltas)
```

特点：

- `x_current` 永远在 softmax 外，AttnRes 不能删除主路径；
- 只路由允许的 delta，不路由累计 hidden state；
- 不增加 entropy、balance 或 route imitation loss；
- 由现有 action、event、JEPA 和 flow loss 自然回传；
- 用于确认路由是否带来动作级增益，以及是否产生捷径。

### 4.2 Replacement 模式：验证后的整体替换

在一个角色组内部，保存角色入口 `x_base` 和各子层 delta：

```text
delta_i = block_i(routed_input_i) - routed_input_i
routed_input_l = x_base + DeltaAttnRes(delta_1 ... delta_(l-1))
```

这时由 Delta AttnRes 替代组内“所有 delta 固定相加”的累计方式，但以下主干仍然受保护：

- role entry/base carrier；
- late raw detail 的独立空间读取；
- noisy action state；
- clean semantic seed；
- world 的 anchor/camera 写权限；
- 最终 action velocity head。

只有 Bridge 模式通过结构、梯度和动作干预验收后，才能进入 Replacement 模式。

## 5. 统一 AttnRes 单元

建议未来实现一个通用但带 schema 的单元，而不是为每个位置临时写不同注意力：

```text
RoleDeltaAttnRes(
    query,
    delta_values,
    route_keys,
    source_role,
    token_schema,
    legal_source_mask,
)
```

### 5.1 Key 与 value

- value：完整 hidden dimension 的真实 delta；
- key/query：低秩路由空间，初始建议研究 `r=16` 和 `r=32`；
- key 必须包含 role、depth、horizon/camera 等必要身份；
- 不把 value 的幅度同时用作路由优势，避免强范数来源自动获胜；
- 路由计算使用 FP32 logits，value accumulation 返回模型 dtype。

### 5.2 delta 定义

每个 delta 必须在明确边界定义：

```text
delta = boundary_output - boundary_input
```

不能用“最终 hidden 减初始 seed”代替所有中间 delta，否则无法区分是哪一层、哪一种写入产生了信息。

world delta 在写入结构化之后保存，即保存 `[anchor, camera]` residual，而不是保存广播后的 64 份重复 xy value。

### 5.3 路由权限

权限由静态 schema mask 决定，不由 loss 学习：

```text
grounding query -> grounding deltas only
world query     -> grounding summaries + earlier world deltas
policy query    -> world deltas + earlier policy deltas
bottom action   -> policy-approved deltas only
```

raw detail 不作为任意深度历史状态混入，而通过独立 protected detail lane 进入 policy。

### 5.4 梯度

第一版不使用：

- straight-through estimator；
- 手写替代梯度；
- route entropy loss；
- 强制均匀或强制 one-hot loss；
- 根据 execution cost 调制路由；
- detach 后再人为补梯度。

梯度沿真实前向路径自然通过 query、key、value 和被选择的上游 block。

## 6. 各候选位置的边界与方案

### A. Grounding 组内部：G1→G2→G3

优先级：中；在边界方案稳定后启用。

当前问题：

- 三个 grounding block 的 residual 固定累计；
- 每层都具有相同角色，但可能分别偏向 goal/history、DINO/flow 和 grounding refinement；
- 当前日志只有整个 grounding group 的 gradient，无法知道三个 delta 是否冗余。

允许来源：

- grounding entry；
- 先前 grounding block 的同 schema delta；
- goal/action-history 只作为 query context，不复制成额外 value 捷径。

token schema：

- clean context 按原 slice；
- rollout 保持 `[anchor, camera, xy]`；
- 不跨 token 类型混合 residual value。

第一版结构：

```text
G1: baseline
G2 query -> {delta_G1}
G3 query -> {delta_G1, delta_G2}
```

不建议第一版替换 G block 内部的 visual cross-attention。它负责空间/视觉读取，与深度路由不是同一维度。

### B. Grounding→World 边界

优先级：高。

目标：

- 让 world 显式选择三个 grounding block 中哪些依据用于每个 anchor/camera；
- 避免把最终 cumulative grounding hidden 当作唯一、不可解释的入口；
- 保留高频 raw detail 的独立 late lane，world 不接管局部细节。

query：

```text
[batch, anchor, camera, route_dim]
```

value bank：

```text
delta_G1_summary
delta_G2_summary
delta_G3_summary
```

grounding delta 的 rollout 部分先在 xy 内池化为 `[anchor, camera]` summary，专供 world 组织。原始 xy detail 继续留在 late raw detail bank，不因池化而丢失。

输出：

```text
world_entry = current_world_entry + routed_grounding_delta
```

禁止：

- world query 直接读取 future teacher target；
- world value 使用 noisy trajectory；
- 将 routed result 写回 xy-specific residual；
- 用一个全局向量覆盖不同 camera/anchor。

### C. World 组内部：W1→W2→W3

优先级：中高。

world block 已经强制输出 `[anchor, camera]` delta，因此它是最干净的 role-local Delta AttnRes 候选。

结构：

```text
W2 query -> {delta_W1}
W3 query -> {delta_W1, delta_W2}
```

query 按 anchor/camera 独立，不能先把全部 anchor 平均成一个 global query。时间邻近性可以进入 key，但不能把所有 horizon 合并成同一 softmax item。

未来完整替换时：

```text
world_state_l = world_entry + DeltaAttnRes(delta_W1 ... delta_W(l-1))
```

world entry 始终在 softmax 外。

### D. World→Policy 边界

优先级：最高。

这是 V102 最值得首先实验的位置，因为当前边界同时承担：

- world temporal/camera organization；
- horizon/basis trajectory query；
- late raw-detail spatial read；
- policy 两块的动作写入。

query：

```text
[batch, action_horizon, action_basis, route_dim]
```

world value：

```text
align_to_horizon(delta_W1)
align_to_horizon(delta_W2)
align_to_horizon(delta_W3)
```

推荐使用两个互不竞争的 lane：

```text
world_context = AttnRes(world_delta_bank)
world_query = QueryProj(world_context)
detail_context = SoftAddressLattice(
    observation_bank,
    grounding_address_state,
    world_query,
)
policy_entry = trajectory + world_context + fixed_detail_scale * detail_context
```

world representation 和 detail value 不进入一个 source-survival softmax；
world 作为 query 完全参与 coarse/fine spatial posterior。这样 world 不会把
detail value 当作可删除的竞争来源，却能与 JEPA、grounding 和 flow 一起决定
最终读取位置。

late detail 的空间注意力升级为：

```text
time/basis/camera query
    -> same-camera soft coarse chart
    -> multi-slot continuous high-resolution candidates
    -> fine detail
```

AttnRes 只决定 world depth contribution；地址格决定 xy/candidate posterior。
8×8 cell 不具有固定 high-resolution patch ownership，并允许跨 cell、多峰和
uncertainty-dependent 搜索。

### E. Policy 组内部：P1→P2

优先级：低到中。

只有两个 policy block，完整 AttnRes 的收益空间有限。第一版仅记录：

- `delta_P1`；
- P2 对 `delta_P1` 的使用；
- P1/P2 分别对 world/detail 的动作级敏感性。

若以后 policy block 增加到三层以上，再使用 role-local Delta AttnRes。当前不应为了使用 AttnRes 而增加网络深度。

### F. Late raw-detail 写入

优先级：高，但只能改“写入组织”，不能替换空间读取。

当前代码直接执行：

```text
trajectory = trajectory + fixed_scale * detail_context
```

未来方案以软多分辨率地址格为准：

1. 缓存 observation-only maps、low-rank keys、flow geometry，不缓存已经
   action/world-conditioned 的最终 posterior；
2. 每个 8×8 cell 保留多个 soft address slot 和连续高分辨率候选；
3. G1/G2/G3 依次形成 alignment、flow rectification 和 coordinate
   canonicalization update；
4. W1/W2/W3 只产生 horizon/camera query，不获得 xy write 权限；
5. P1/P2 在 read time 形成最终 detail，首次写入仍为 protected addition；
6. 候选 raw value 先在小通道空间加权，再投影到 hidden size，避免保存
   `candidate_count × hidden_size` 的巨大 value bank。

禁止：

- 让 detail 与 pooled DINO/world 在同一个 amplitude router 中竞争；
- 用 softmax route mass 作为 detail 的唯一生存通路；
- 让 action token 参与 detail bank 编译；
- 把 detail 提前融合回 world xy chart。
- 把 8×8 cell 硬绑定到一个固定 high-resolution patch；
- 在 G/W query 参与前提前把局部候选压成一个不可恢复的 detail vector。

### G. 顶层 policy workspace→底层 Evidence MMDiT

优先级：最高。

当前是一个固定的二分支方差保持融合：

```text
action = sqrt(0.5) * action_seed + sqrt(0.5) * normalized_policy_workspace
```

它解决了历史 `0.10` 瓶颈，但把 P1/P2 及其 world/detail 来源压成了一个最终 workspace，底层无法判断信息来自哪里。

Bridge 方案：

```text
action_seed = noisy_state + semantic_seed + horizon_position
policy_context = AttnRes({delta_P1, delta_P2, approved_W_to_P_delta})
action = protected_action_seed + fixed_scale * policy_context
```

完整替换方案：

- Evidence adapter 接收 typed policy delta bank；
- 每个 action horizon query 独立选择 P1/P2/world-approved delta；
- noisy action state 和 semantic seed 永远保持直接路径；
- late detail 只有在已经进入 policy-approved delta 后才能进入底层。

这比把 noisy action、semantic seed、world 和 raw detail 全部放入一个 softmax 更安全。

### H. EvidenceConditionOrganizer 的 layer scan

优先级：高。

当前 organizer：

1. 从 layer contracts 构造 layer rows；
2. 用 GRUCell 顺序扫描；
3. 与 clean intent attention 相加；
4. 生成唯一 global `z`。

严格角色模式下，最终 decoder 只收到最后两个 policy contracts，因此这个 depth scan 实际没有看到 3-3-2 的完整组织过程。

建议替换为两个分离对象：

#### H1. Clean semantic AttnRes

- query：trainable intent query + time；
- selector key：角色边界 summary；
- value：clean intent memory；
- 输出：global `z`；
- policy/noisy delta 可以提供 selector geometry，但不能直接成为 global semantic value。

这延续当前 `layer selector -> clean intent value` 的防捷径原则。

#### H2. Action evidence AttnRes

- query：每个 action horizon token；
- key/value：policy-approved role delta；
- 输出：供 MMDiT evidence cross-read；
- 不写回 global `z`。

因此 global semantic organization 和 action-specific evidence retrieval 不会再次合并成一个隐蔽旁路。

### I. 底层三层 TimeDomain MMDiT

优先级：中，最后实施。

每个 MMDiT block 当前包含：

- causal action self-attention update；
- typed evidence cross-attention update；
- FFN update；
- execution gate/capacity 作用于实际 forward residual。

不应直接把三个 update 混成一个 AttnRes bank。若以后替换，至少拆成：

```text
self_delta_bank
evidence_delta_bank
ffn_delta_bank
```

每个后续 MMDiT block 可以选择以前的同类型 delta；evidence delta 还应保留 source attribution。执行控制器的 gate 必须继续作用于最终实际写入，不能退回 audit-only。

第一版不改变底层 MMDiT，因为它同时涉及执行控制和 ODE 多步采样，回归面最大。

## 7. 明确不替换的位置

| 位置 | 决策 | 原因 |
|---|---|---|
| DINO/raw/flow 空间相关性搜索 | 不替换 | 它解决空间地址，不是深度选择 |
| LateRawDetailPolicyReader 的相机内 xy attention | 不替换 | 需要保持精细空间带宽和相机所有权 |
| world anchor/camera write structuring | 不替换 | 这是角色契约，不是普通残差缺陷 |
| noisy action state 直达 action stream | 不替换 | flow matching 必须知道当前 `x_t` |
| clean semantic seed 直达 action stream | 不替换 | 防止所有语义依赖单个可塌缩 route |
| action/event/motion heads | 不替换 | 它们是任务读出，不是信息组织 |
| execution value/capacity/dwell controller | 暂不替换 | 与深度路由的目标不同，且具有独立 forward 控制语义 |
| 所有 block 内部 FFN residual | 第一阶段不替换 | 八层网络不深，先验证角色边界收益 |

## 8. 初始化与防塌缩

### 8.1 不使用硬门控

AttnRes 权重使用连续 softmax；合法来源由静态 role mask 限制。部署时不把 route 强行 argmax。

### 8.2 不让必要路径参加生存竞争

以下路径位于 AttnRes 外：

- current/base carrier；
- raw detail 首次 protected write；
- semantic seed；
- noisy action state。

这样即使路由早期不成熟，也不会切断关键梯度。

### 8.3 不路由累计 state

默认路由 delta。累计 state 仅可作为 query，不作为 bank value。这样降低历史表示重复导致的近均匀路由。

### 8.4 防 attention sink

不通过额外 loss 强制熵，而通过诊断和结构限制发现 sink：

- 每个 route 的 max probability、normalized entropy；
- 每个 source/role 的平均 mass；
- batch/episode 间 route variance；
- route key/value norm；
- routed update RMS 和 infinity norm；
- 单个来源长期占据绝大多数 mass 的持续时间；
- zero/shuffle 某个 delta 后的 action change。

若出现 sink，优先调整 key normalization、low-rank dimension、role grouping 或加入显式 null/current slot；不要先加 route balance loss。

## 9. 必须新增的日志

每个 AttnRes site 至少记录：

```text
attnres_<site>_entropy
attnres_<site>_max
attnres_<site>_source_mass_<role_or_depth>
attnres_<site>_delta_norm_<source>
attnres_<site>_update_norm
attnres_<site>_carrier_ratio
attnres_<site>_key_rms
attnres_<site>_value_rms
attnres_<site>_activation_rms
grad_attnres_<site>
```

对 `[time, basis]`、`[anchor, camera]` 或 `[camera, xy]` 路由，还需记录：

- horizon 间 route 差异；
- camera 间 route 差异；
- basis 内 route 差异；
- route 在 ODE steps 之间是否持续存在；
- prediction horizon adjacent cosine 与 target adjacent cosine 的差值。

不要只记录平均 mass。平均值可能掩盖不同样本上完全不同的选择。

## 10. 动作级干预

每个新 route 必须有对应 eval-only probe：

```text
baseline
route_delta_zero
route_delta_episode_shuffle
route_depth_permute
route_uniform
route_latest_only
```

报告：

- action RMSE/MSE 的成对变化；
- action delta RMSE；
- arm/gripper 分开；
- decoded event precision/recall/F1；
- horizon bands 1-4、5-12、13-24；
- episode-cluster bootstrap 置信区间；
- representation boundary 的实际变化量。

只有“boundary representation 确实改变，且 action 输出显著改变”才能证明 route 到达部署路径。仅有非零梯度或非均匀 attention 不足以证明有用。

## 11. 分阶段实验路线

### Phase 0：完成 V102 基线

在修改 AttnRes 前取得：

- 至少一个完整 epoch validation；
- V102 sample-time world/late-detail metrics；
- late-detail zero/shuffle action probe；
- world anchor shuffle action probe；
- horizon target/prediction differentiation；
- gripper event closure。

### Phase 1：只暴露 delta，不改变 forward

记录 G1–G3、W1–W3、P1–P2 的结构化 delta：

- 检查 shape 和所有权；
- 测量 delta 相关性、冗余度和范数；
- 验证 world delta 在 xy 内仍为常数；
- 验证 detail bank 不含 action/world 依赖。

### Phase 2：World→Policy Bridge AttnRes

只改 D 位置：

- world delta 按 horizon/basis 检索；
- late detail 仍走 protected lane；
- 底层和 organizer 不变。

这是第一组因果最清晰的实验。

### Phase 3：Grounding→World Bridge AttnRes

只改 B 位置：

- world 按 anchor/camera 选择 grounding summaries；
- 继续禁止 world xy write；
- 与 Phase 2 组合前先做单独对照。

### Phase 4：Policy workspace→MMDiT

将单一 final workspace 替换成 typed policy delta retrieval，但 noisy action 和 semantic seed 保持直接路径。

### Phase 5：Organizer

用 clean semantic AttnRes 替换 GRU layer scan，并把 action evidence retrieval 与 global `z` 分开。

### Phase 6：角色组内部整体替换

依次启用：

1. W1–W3 Delta AttnRes；
2. G1–G3 Delta AttnRes；
3. P1–P2（只有前面证明 P1/P2 冗余或筛选价值时）。

### Phase 7：底层 MMDiT

最后研究 self/evidence/FFN 三种 delta 的分类型 AttnRes。不得与 execution controller 重构同时进行。

## 12. 验收标准

### 12.1 结构验收

- world xy residual 继续保持数值零；
- world anchor/camera residual 非零；
- late detail token count 等于 `camera_count * reader_grid^2`；
- sampling 的每个 ODE step 都能访问 observation-only detail cache；
- policy 不得直接读取全局 raw/DINO bank；
- global `z` 不得读取 noisy action value；
- role mask 不允许跨权限写入；
- flags off 时与 V102 baseline 数值等价。

### 12.2 优化验收

- 所有 active AttnRes route 有自然非零梯度；
- route 不长期全均匀，也不长期被单个 source 跨全部样本占满；
- routed update 相对 carrier 的量级稳定；
- activation RMS、infinity norm 和 preclip gradient 不持续增长；
- 不新增单独的 route loss；
- loss ledger 能精确重建 backward scalar。

### 12.3 任务验收

- validation full/first/first8/tail RMSE 不回退；
- 1-4、5-12、13-24 三个 horizon band 不因平均值掩盖回退；
- arm 与 gripper 分开比较；
- decoded gripper event F1 不回退；
- route zero/shuffle 对 action 的影响具有 episode-level 统计证据；
- detail route 的动作影响不能只来自 pooled world/semantic 替代。

## 13. Checkpoint 与配置

未来所有 AttnRes 配置默认关闭：

```text
role_attnres_enabled=0
role_attnres_mode=bridge|replace
role_attnres_key_dim=16|32
role_attnres_ground_to_world=0
role_attnres_world_to_policy=0
role_attnres_policy_to_mmdit=0
role_attnres_clean_organizer=0
role_attnres_bottom_mmdit=0
```

要求：

- flags off 时旧 checkpoint 严格兼容；
- 新模块 missing keys 必须在显式迁移报告中列出；
- 不静默加载形状相近但语义不同的旧参数；
- 不用历史 Stage1 checkpoint 初始化新的路由，除非另有受控迁移实验；
- 每次只启用一个新 boundary，组合实验必须有前序单边界结果。

## 14. 当前 V102 早期日志对 AttnRes 的启示

附件日志覆盖 epoch 1 的 batch 20–760，共 38 个 console sampling points，没有完整 epoch validation。

已观察到：

- world spatial residual 约为 `1e-6`，anchor/camera residual 非零并增大，证明 V102 world 写权限生效；
- late-detail attention 从接近均匀逐渐集中，但 update/trajectory ratio 从约 `0.03` 降到约 `0.004–0.005`；
- late-detail reader、raw address reader 和 policy blocks 都有非零梯度；
- moving-region warp gain 后期为正，但 correlation entropy 仍高；
- policy block gradient 明显大于 world block gradient；
- top-policy workspace update 仍是大尺度固定融合来源；
- horizon adjacent cosine 升高，但当前日志没有 target adjacent cosine，不能判断这是正确共享还是时间尺度被抹平。

因此：

1. world ownership 已经修正，不应被 AttnRes 破坏；
2. world→policy 是第一优先级，因为它正处于“大 world residual、较小 late-detail write、强 policy gradient”的交界；
3. AttnRes 不能让 world 和 detail 在同一 softmax 中互相淘汰；
4. organizer 只看 terminal policy rows，确实存在用角色边界检索进行整体替换的空间；
5. 没有 validation 和 action intervention 前，不能声称 AttnRes 已经是 V102 当前问题的确定修复。

## 15. 最终目标结构

```text
Goal Tokens + Action History
              │
RGB/DINO + learned flow ── spatial reader ── protected detail bank
              │
              ▼
        G1 ─ G2 ─ G3
              │
      Ground→World Delta AttnRes
              │
        W1 ─ W2 ─ W3
              │
      World→Policy Delta AttnRes
              ├──────── protected late-detail spatial read
              │
        P1 ─────── P2
              │
      typed policy delta bank
              │
      clean semantic organizer
              │
     bottom Evidence MMDiT × 3
              │
       native action trajectory
```

这里 AttnRes 负责“从已经形成的角色增量中选择什么”，flow/raw reader 负责“从图像哪里取得高精度信息”，JEPA 负责“未来表征应该朝哪里变化”，底层 MMDiT 负责“如何在当前 flow state 上形成动作”。四者职责不能再次压成一个模糊的统一路由器。
