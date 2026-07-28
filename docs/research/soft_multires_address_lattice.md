# Soft Multi-Resolution Address Lattice

状态：ClearVLA 未来空间地址机制的详细设计记录，不对应当前已实现版本。

关联文档：[3-3-2 AttnRes 结构迁移计划](attnres_332_migration_plan.md)

## 1. 核心结论

8×8 结构不能被解释成 64 个与高分辨率图像区域硬性一一对应的指针。

每个 8×8 cell 应是一个可学习的地址查询单元，拥有：

- 多个软地址 slot；
- 对整个 coarse semantic chart 的自主选择能力；
- 对高分辨率局部候选的连续可微读取；
- learned flow 提供的几何先验；
- JEPA、grounding 和 world DiT 提供的语义、时间与任务条件；
- 地址均值、方差、多峰程度和内容值，而不是单个硬坐标。

目标结构是：

```text
observation-only multi-resolution bank
                │
                ├── raw 84×84 / mid 42×42
                ├── DINO 16×16
                ├── learned flow / confidence / uncertainty
                └── camera / coordinate identity
                                │
                     8×8 soft address lattice
                                │
            G1 alignment → G2 rectification → G3 canonicalization
                                │
                 W1/W2/W3 horizon-conditioned reweighting
                                │
                     P1/P2 high-resolution readout
```

learned flow 是 prior，不是最终地址；8×8 cell 是 query，不是硬指针；最终地址是条件化概率分布。

## 2. 为什么不能使用硬对应

固定的 8×8→高分辨率映射会隐含以下不成立的假设：

1. 每个 coarse cell 只对应一个局部高分辨率区域；
2. flow 的误差不会跨越 coarse cell；
3. 一个 cell 只包含一个任务相关对象；
4. 遮挡、反光、夹爪和目标物不会产生多峰对应；
5. 不同 horizon 对同一 coarse cell 的细节需求相同；
6. camera 与 source/current 坐标可以通过一次固定变换完全对齐。

真实场景中，一个 coarse query 可能需要：

- 跳到相邻甚至更远的 cell；
- 同时保留两个候选位置；
- 在不同 future horizon 选择不同细节；
- 在 flow 不可靠时依赖 DINO/raw 内容；
- 在 DINO 语义粗糙时依赖 raw/flow 几何；
- 在遮挡时保留高方差而不是强行选点。

因此地址必须是 soft、multi-slot、可跨格和连续的。

## 3. 与当前 V102 的差别

当前 V102 的 observation-only 思路应保留，但候选压缩发生得过早：

```text
flow-centered high-resolution candidates
                ↓
raw/source matching 提前压成每个 8×8 cell 一个 detail vector
                ↓
world/policy 只能在 8×8 成品 token 之间选择
```

新机制改为：

```text
observation-only maps + candidate geometry 只编译、不提前做最终压缩
                ↓
G1/G2/G3 形成 soft address state
                ↓
W1/W2/W3 形成 horizon-specific address posterior
                ↓
P1/P2 才读取高分辨率 value
```

这保留 V102 的：

- action-independent observation bank；
- 多 ODE step 可复用；
- camera ownership；
- world anchor/camera-only write；
- late high-resolution read；

同时避免让弱 flow 独自决定早期高分辨率候选。

## 4. 主要数据结构

### 4.1 AddressObservationBank

只由 observation 编译，可缓存：

```text
AddressObservationBank:
    dino_keys             [B, C, 16, 16, Rk]
    dino_values           [B, C, 16, 16, Rd]
    raw_high_features     [B, C, 84, 84, Cr]
    raw_mid_features      [B, C, 42, 42, Cm]
    raw_coarse_features   [B, C, 8,  8,  Cc]
    flow_forward          [B, C, 84, 84, 2]
    flow_backward         [B, C, 84, 84, 2]
    confidence            [B, C, 84, 84, 1]
    uncertainty           [B, C, 84, 84, 1]
    occlusion             [B, C, 84, 84, 1]
    correlation_features  [B, C, 84, 84, Rc]
    camera_identity
    coordinate_identity
```

其中：

- `Rk` 是低秩路由维度，建议先研究 32/64；
- raw value 保持较小原始通道，不提前全部投影到 hidden size；
- future teacher target 不进入该 bank；
- noisy action、proposal、policy trajectory 不进入该 bank。

### 4.2 SoftAddressState

每个 8×8 cell 拥有多个地址 slot：

```text
SoftAddressState:
    coarse_logits      [B, C, 8, 8, M, 16, 16]
    coarse_probs       [B, C, 8, 8, M, 16, 16]
    coarse_centers     [B, C, 8, 8, M, 2]
    coarse_variance    [B, C, 8, 8, M, 2]
    fine_offsets       [B, C, 8, 8, M, K, 2]
    fine_logits        [B, C, 8, 8, M, K]
    fine_probs         [B, C, 8, 8, M, K]
    address_content    [B, C, 8, 8, M, H]
    address_geometry   [B, C, 8, 8, M, G]
```

`M` 是软地址 slot 数，不是硬 top-k。建议从 `M=4` 研究，但不把它写成永久架构常数。

`K` 是每个 slot 的连续局部采样数，可复用当前半径 3 的 7×7、即 49 个采样位置作为初始对照。

### 4.3 HorizonAddressState

world 之后增加 horizon/basis 条件：

```text
HorizonAddressState:
    logits   [B, T, basis, C, 8, 8, M, K]
    probs    [B, T, basis, C, 8, 8, M, K]
    content  [B, T, basis, C, 8, 8, H]
    geometry [B, T, basis, C, 8, 8, G]
```

实际实现不必物化完整大张量，可以分块计算；上面是语义 contract。

## 5. Coarse 自主选择

### 5.1 每个 cell 是 query，不是固定区域

对 source 8×8 cell \((i,j)\) 和地址 slot \(m\)，生成 query：

\[
q_{ijm} = Q_m(
    \text{clean grounding state}_{ij},
    \text{source DINO}_{ij},
    \text{goal/history}
)
\]

它可以对同一 camera 的完整 16×16 DINO chart 做 soft attention：

\[
L^{coarse}_{ijm}(u)
=
q_{ijm}^{T}k_{\text{DINO}}(u)
+ b_{\text{flow}}(u)
+ b_{\text{coord}}(u)
+ b_{\text{camera}}(u)
\]

这里不使用 argmax、top-k 或 straight-through estimator。

### 5.2 Flow 只提供几何偏置

将 source cell 中心通过 learned flow 映射为 proposal center：

\[
c^{flow}_{ij}=S(i,j)+F_{ij}
\]

flow prior 可以写成：

\[
b_{\text{flow}}(u)
=-\frac{\|S(u)-c^{flow}_{ij}\|^2}
        {2(\sigma^{flow}_{ij})^2+\epsilon}
\]

但它不是硬 mask：

- flow 可靠时，附近候选获得更强先验；
- flow 不可靠时，\(\sigma\) 增大，分布变宽；
- DINO/grounding 内容可以选择远离 flow proposal 的位置；
- coarse attention 始终保留跨 cell 纠错能力。

### 5.3 多 slot 而不是单个期望点

每个 slot 使用独立 query projection/slot identity，形成多个 coarse posterior：

\[
p_{ijm}(u)=\operatorname{softmax}_u L^{coarse}_{ijm}(u)
\]

每个 slot 输出软中心和方差：

\[
\mu_{ijm}=\sum_u p_{ijm}(u)S(u)
\]

\[
\Sigma_{ijm}=\sum_u p_{ijm}(u)
             (S(u)-\mu_{ijm})^2
\]

不使用单一 posterior 的均值代表多峰分布；多个 slot 分别保留可能的峰。

第一版不增加 slot diversity loss。先通过独立 slot identity、不同 query projection 和自然任务梯度观察是否发生 slot collapse。

## 6. Fine 高分辨率自主读取

### 6.1 连续候选坐标

以每个 coarse slot 的 \(\mu_{ijm}\) 为中心，生成连续局部候选：

\[
x_{ijmk}
=
\mu_{ijm}
+ r_{ijm}\delta_k
+ d_{ijm}
\]

其中：

- \(\delta_k\) 是规范化局部采样格；
- \(r_{ijm}\) 由 coarse variance、flow uncertainty 和 occlusion 共同决定；
- \(d_{ijm}\) 是 query-dependent bounded residual correction；
- 所有坐标使用连续 `grid_sample`，不要求 84/8 是整数比例；
- 相邻 8×8 cell 的候选范围允许重叠。

### 6.2 Fine key

每个连续坐标读取多尺度 key：

\[
k^{fine}(x)
=
\phi(
    raw_{84}(x),
    raw_{42}(x),
    DINO_{16}(x),
    correlation(x),
    confidence(x),
    coordinate(x)
)
\]

fine logits：

\[
L^{fine}_{ijmk}
=
q^{fine}_{ijm}{}^T k^{fine}(x_{ijmk})
+ b_{\text{local-flow}}
+ b_{\text{valid}}
\]

fine posterior：

\[
\pi_{ijmk}
=
\operatorname{softmax}_k L^{fine}_{ijmk}
\]

### 6.3 Fine value

高频内容：

\[
v_{ijm}
=
\sum_k \pi_{ijmk}v_{raw}(x_{ijmk})
\]

最终投影：

\[
a_{ijm}=W_v(v_{ijm})
\]

value projection 应在加权聚合之后执行。若 `W_v` 是线性的，这与“每个候选先投影到 hidden size、再加权”数学等价，却显著减少显存。

## 7. 8×8 与高分辨率的真实关系

新结构不规定：

```text
cell (i,j) -> high-resolution fixed patch (i,j)
```

而是：

```text
cell (i,j)
    -> M 个 coarse soft distributions
    -> M 个连续高分辨率局部候选场
    -> M 个地址内容与几何状态
```

因此允许：

- 跨 cell；
- 多峰；
- 相邻 cell 重叠；
- flow proposal 被内容纠正；
- 同一 cell 在不同 horizon 选择不同位置；
- 地址不确定时保持宽分布；
- 多 camera 保留独立空间身份。

## 8. 3-3-2 中前三块的明确职责

前三块共享同一个 observation bank 和 persistent address state，但各自拥有不同的输入所有权、状态字段和 update contract。它们不是三次相同读取。确定性的连续坐标、尺度映射、camera identity、validity 和候选采样几何在进入 G1 前建立；这是 coordinate scaffold，不是学习后的 canonicalization。

### 8.1 G1：Candidate hypothesis / alignment

主要输入：

- source/current DINO；
- raw coarse/mid appearance；
- camera/xy identity；
- clean state/history；
- 每个 slot 独立的 query identity。

主要更新：

- coarse semantic logits；
- source/current coarse correspondence；
- slot-specific coarse posterior。

输出：

```text
delta_address_G1
alignment_summary
```

G1 保持每个 camera 独立，不做 camera value 融合；保持多个 slot，不输出唯一 expected coordinate。Goal/phase 可以在以后作为 selector query，但不得进入 observation value。

### 8.2 G2：Rectification

主要输入：

- G1 posterior；
- learned flow；
- confidence/uncertainty；
- forward/backward cycle；
- raw correlation；
- occlusion。

主要更新：

- flow prior strength的局部、输入相关解释；
- coarse variance；
- fine radius；
- bounded continuous residual offset；
- invalid/occluded candidate probability。

输出：

```text
delta_address_G2
rectified_geometry
```

### 8.3 G3：Canonicalization

主要输入：

- G1/G2 address state；
- source/current/camera coordinate identity。

主要更新：

- source/current chart 统一；
- horizon-independent clean address basis；
- world 可读取的 anchor/camera summary；
- policy 后续读取所需的完整 candidate geometry。

输出两个并行接口：

```text
clean_spatial_address_bank [camera, 8, 8, slot, candidate]
grounding_anchor_summary   [anchor, camera]
```

不能只保留第二个接口。

G3 的 canonicalization 是学习后的候选状态编译，不是第一次坐标定义，也不是 high-resolution value read。G3 禁止把 `camera × source cell × slot × fine candidate` 通过一次 softmax 压成 rollout value。G3 后仍保留完整 bank；W 产生 horizon posterior，P 才执行最终 value read。

### 8.4 为什么不整体反转 G1/G2/G3

- 没有 G1 候选 posterior，G2 没有可整流对象；
- 没有 G2 几何与不确定性修正，学习后的 canonicalization 会退化为弱 flow 或固定坐标假设；
- 如果最后才建立 alignment，W 接收的始终是未 grounded 状态。

需要前移的是 deterministic coordinate scaffold；需要后移的是跨 camera 的任务相关选择和 high-resolution value read。学习后的 canonical compilation 仍位于 G3。

### 8.5 W/P 的后续所有权

```text
Pre-G: coordinate/camera/scale scaffold
G1:    multi-slot coarse hypotheses
G2:    flow/DINO/raw geometric rectification
G3:    canonical address basis + typed grounding summaries
W1:    near-horizon posterior update
W2:    mid-horizon consequence posterior update
W3:    far-horizon/phase posterior update
P1/P2: action/goal/phase-conditioned final fine-value read
```

Implementation boundary: a complete-chart G1 posterior can move a mode beyond
the old coarse compiler's local sampling window. Under the V109 flag, the
observation bank therefore retains dense projected source/target raw-key
charts, a target DINO-key chart, and the narrow raw-detail chart. G2
differentiably rematerializes candidates around its corrected centres while
retaining every camera/source-cell/slot/fine axis. It does not aggregate
values. The first query-dependent high-resolution value aggregation remains
strictly at W->P. Reusing only the old-centre candidates would let G1/G2 change
telemetry and priors without actually reaching the new high-resolution
location, and is not an accepted implementation.

At W3->P1, W scores the canonical G3 source basis before P runs. The same
compatibility tensor yields (a) the target-cell marginal supervised by the
weak horizon-relevance objective and (b) a bounded source-cell/slot prior used
by P's only high-resolution value read. These are selector-only marginals, not
two value readers: raw values remain unaggregated until P and are aggregated
exactly once.

Observation bank 始终 horizon independent；W posterior 始终 query owned。不得缓存 action/world-conditioned posterior，也不得让 W/P query 反向污染 observation values。

## 9. Clean address stream

前三块的 address stream 必须与 noisy trajectory 分离。

允许读取：

- observation；
- state/state history；
- executed action history；
- Goal Tokens；
- camera/coordinate identity；
- JEPA online prediction/query。

禁止读取：

- 当前 flow-matching noisy action `x_t`；
- future teacher DINO target；
- teacher future change；
- policy trajectory；
- action proposal 中可能携带的当前预测捷径。

推荐在 canvas 中增加显式 `address` slice，配套 directed attention mask，而不是从已经 action-conditioned 的 rollout token 中临时抽取 clean query。

## 10. World blocks 的参与方式

W1–W3 保留完整的 `[anchor,camera,xy]` coarse world chart。旧的
`write schema = [anchor,camera]` 会在精细地址读取前抹掉空间差异，已经被
V103 的 M17/M18 修复取代。W 可以写 coarse spatial residual，并读取 G3
address summary；但它仍不能聚合或改写高分辨率 raw values。

```text
q_world[anchor, camera, xy]
```

world 对地址的作用是增加 query/logit update：

\[
L^{world}_{h,ijmk}
=
L^{G3}_{ijmk}
+ \Delta L_{W1}
+ \Delta L_{W2}
+ \Delta L_{W3}
\]

这样 world 参与最终地址选择并拥有 query-owned coarse spatial state，
但没有获得 observation bank 或高分辨率 value 的所有权。

不同 horizon 的 query 独立，不能先平均所有 anchor 再生成一个共享空间地址。

## 11. Policy blocks 的读取

P1/P2 输入：

- noisy action trajectory；
- world horizon query；
- G3 clean address bank；
- horizon-specific address posterior；
- goal/action history；
- policy workspace。

P1/P2 在 read time 形成最终 detail：

```text
detail_context[T, basis, camera]
```

各 camera 先独立读取。第一版沿用固定方差保持的 camera combination，避免新增可塌缩的全局 camera router。

detail context 作为受保护的 additive lane 进入 trajectory，不能与 pooled world/DINO 在一个 source survival softmax 中竞争。

## 12. JEPA 的角色

JEPA 提供：

- online future query；
- 不同真实 horizon 的表示方向；
- change-aware grounding；
- 对 G3/W blocks 的未来表征监督。

JEPA 不提供：

- teacher future feature 作为 forward 输入；
- 确定性的 future coordinate；
- 对 raw detail 的硬 mask。

未来 DINO target 只参与 loss，不参与 address query。

## 13. Learned flow 的角色

learned flow 负责：

- source→current 的局部 transport proposal；
- coarse prior center；
- fine sampling 的初始几何；
- uncertainty/search-radius 依据；
- motion/cycle 的连续监督对象。

learned flow 不负责：

- 独自输出最终动作相关空间地址；
- 决定 task relevance；
- 决定不同 horizon 应读取什么；
- 把 84×84 候选提前压成不可恢复的 8×8 token；
- 通过硬 window 排除所有远距离纠错。

以后日志中应把：

```text
flow proposal quality
```

和：

```text
joint address posterior quality
```

分开命名，不能再统称为“flow address quality”。

## 14. 与 AttnRes 的接口

AttnRes 不处理 xy candidate softmax。它处理深度与角色 delta：

```text
delta_G1 / delta_G2 / delta_G3
              ↓
Grounding Delta AttnRes
              ↓
address query/update
```

```text
delta_W1 / delta_W2 / delta_W3
              ↓
World Delta AttnRes
              ↓
horizon-conditioned query/update
```

空间地址仍由 Soft Address Lattice 的 coarse/fine posterior 形成。

AttnRes 与地址格的组合原则：

- AttnRes 选择“使用哪一层组织信息”；
- address lattice 选择“在空间哪里读取”；
- value lane 保留 raw/DINO 高精度内容；
- flow 始终作为几何 proposal，而不是深度路由 value。

## 15. 自主选择与防捷径

### 15.1 禁止硬选择

第一版不使用：

- argmax coordinate；
- hard top-k；
- straight-through estimator；
- non-differentiable crop；
- 离散 cell ownership；
- 训练时 hard camera selection。

### 15.2 不强迫 flow 赢

不增加“flow route mass 必须大于某阈值”的 loss。

flow 是否有用由：

- raw warp/cycle；
- JEPA；
- action/event；
- zero/shuffle intervention；

共同决定。

但 flow 必须真实改变 proposal geometry，不能只作为一个可被下游完全忽略的标量 feature。

### 15.3 不强迫均匀或尖锐

不增加固定 entropy target。不同场景可能需要：

- 高置信单峰；
- 多峰；
- 遮挡下宽分布；
- horizon-dependent selection。

只记录分布，不人为规定所有样本的理想熵。

### 15.4 防止 pooled fallback 接管

完整低频 base 可以保留为 additive content lane，但不能作为与 high-frequency address lane 竞争的替代路线。

地址机制负责 detail；base 负责完整低频内容。二者固定相加或方差保持融合。

## 16. 计算与显存

### 16.1 不缓存 full-hidden candidate values

禁止缓存：

```text
[B, C, 8, 8, M, K, hidden_size]
```

建议缓存：

```text
low-rank candidate keys
raw candidate coordinates
raw/mid/high feature maps
confidence/uncertainty
```

attention 权重形成后，先在小通道 raw value 上求和，再投影到 hidden。

### 16.2 分块计算

可按以下轴分块：

- camera；
- horizon；
- basis；
- address slot；

不改变 soft posterior 的语义。

### 16.3 ODE 复用

跨 ODE step 缓存：

- AddressObservationBank；
- observation-only key projections；
- coordinate/flow geometry。

每步重新计算：

- action/world-conditioned query；
- horizon-specific logits；
- final detail read。

不能缓存 action-conditioned posterior。

### 16.4 激活检查点

高分辨率 raw pyramid 与 candidate sampling 可继续使用 activation checkpoint，但不得在训练 forward 中误用 no-grad cache。

## 17. 输出与日志

每个阶段至少记录：

### 17.1 Flow proposal

```text
flow_proposal_magnitude
flow_proposal_uncertainty
flow_proposal_warp_gain
flow_proposal_moving_gain
flow_proposal_static_gain
```

### 17.2 Coarse address

```text
address_coarse_entropy
address_coarse_max
address_coarse_variance
address_cross_cell_distance
address_flow_center_distance
address_slot_pair_distance
address_slot_effective_count
```

### 17.3 Fine address

```text
address_fine_entropy
address_fine_max
address_fine_radius
address_fine_offset_norm
address_highres_valid_fraction
address_content_norm
address_geometry_norm
```

### 17.4 角色更新

```text
address_update_G1
address_update_G2
address_update_G3
address_update_W1
address_update_W2
address_update_W3
address_update_P1
address_update_P2
```

### 17.5 梯度

```text
grad_address_coarse_query
grad_address_flow_bias
grad_address_fine_query
grad_address_raw_key
grad_address_dino_key
grad_address_value_projection
grad_grounding_address_stream
grad_world_address_query
grad_policy_detail_read
```

平均值之外，还应保留 episode/horizon/camera 分布。

## 18. 因果探针

必须支持 evaluation-only、同噪声干预：

```text
baseline
flow_zero
flow_episode_shuffle
flow_spatial_shuffle
dino_key_shuffle
raw_value_zero
raw_value_spatial_shuffle
G1_delta_zero/shuffle
G2_delta_zero/shuffle
G3_delta_zero/shuffle
world_query_zero/shuffle
address_posterior_uniform
address_slot_permute
fine_offset_zero
```

每个探针同时报告：

- address representation delta；
- final detail delta；
- action delta RMSE；
- validation MSE/RMSE change；
- arm/gripper 分开；
- event precision/recall/F1；
- horizon bands；
- episode-cluster bootstrap interval。

若 representation 没变，探针无效；若 representation 改变而 action 不变，说明下游忽略或补偿。

## 19. Loss 原则

第一版不新增 address-specific auxiliary loss。

使用现有自然监督：

- raw warp/cycle/smoothness；
- flow uncertainty NLL；
- moving/static identity constraints；
- JEPA future/change；
- action flow matching；
- decoded action/event/motion；
- rollout/transition；
- layer/ownership contracts。

只有当动作级探针证明地址 posterior 有效改变但优化长期不稳定，并且缺失监督可以被明确识别时，才讨论额外 loss。

禁止为日志好看而添加：

- route mass loss；
- fixed entropy loss；
- slot diversity loss；
- flow usage quota；
- hard address imitation。

## 20. 分阶段迁移

### Phase A：只编译 observation bank

- 新建 AddressObservationBank；
- 与当前 V102 输出做数值和 shape 对照；
- 不改变 action forward；
- 验证缓存、dtype、显存和 ODE replay。

### Phase B：Shadow soft lattice

- 运行 coarse/fine posterior；
- 只记录地址和梯度，不写入 trajectory；
- 与 current raw reader 的输出比较；
- 做 flow/DINO/raw intervention。

### Phase C：替换 late raw read

- 使用 soft lattice 读取 detail；
- 仍保持 G/W blocks 原实现；
- detail 通过受保护 fixed-scale lane 写入 policy；
- 对照当前 V102 late reader。

### Phase D：接入 G1/G2/G3

- 增加 clean address stream；
- G1/G2/G3 分别产生 alignment/rectification/canonicalization update；
- 在 G1 前建立只含几何和身份的 deterministic coordinate scaffold；
- G1 保留 per-camera multi-slot hypothesis，G2 保留 rectified modes，G3 只编译 canonical basis，不读取 high-resolution value；
- 禁止 noisy action 进入；
- 验证三个不同干预分别改变 hypothesis、geometry 和 canonical basis，并验证 action/JEPA 自然梯度。

### Phase E：接入 W1/W2/W3

- world 产生 near/mid/far horizon/camera query；
- W 只更新 query-owned posterior/world state，不重写 observation bank；
- 每个 horizon 重加权 G3 soft address basis，并将最终 high-resolution value read 推迟到 P1/P2。

### Phase F：接入 AttnRes

- Grounding Delta AttnRes 组织 G1–G3；
- World Delta AttnRes 组织 W1–W3；
- 不改变空间 softmax 机制；
- 最后再考虑 policy→bottom MMDiT。

### Phase G：移除旧双 reader

只有在以下条件满足后删除旧路径：

- validation 不回退；
- detail action intervention 显著；
- flow/DINO/G/W 各自具有可解释作用；
- ODE 多步 replay 正确；
- 显存和速度可接受；
- flags off 仍可复现 V102。

## 21. 静态验收

实现前后的测试要求：

1. 8×8 cell 不具有固定、离散 high-resolution patch id；
2. coarse attention 覆盖完整同 camera DINO chart；
3. fine coordinates 连续且允许跨 cell；
4. 所有选择使用可微 soft weights；
5. AddressObservationBank 不依赖 noisy action；
6. future teacher target 不进入 forward；
7. world 仍不能写 xy residual；
8. camera identity 在 fine read 完成前不丢失；
9. raw detail 为零时 detail update 精确为零；
10. flags off 时 V102 路径行为不变；
11. BF16 输入下 logits 使用 FP32；
12. candidate 无效区域不会产生 `inf/NaN`；
13. cache 与非 cache 输出一致；
14. action loss 可以自然回传到 G/W query、flow、DINO/raw key 和 value projection；
15. 不存在 detach 后的人工梯度补丁。

## 22. 实验成功标准

成功不等于 attention 更尖锐，也不等于 flow mass 更大。

必须同时满足：

- moving-region flow proposal 比 zero flow 更好；
- joint address 比 flow-only 和 DINO-only 地址更好；
- G/W intervention 能改变 address posterior；
- detail zero/shuffle 能显著改变 action；
- validation horizon 与 gripper/event 不回退；
- 不通过 pooled content 绕过 high-frequency detail；
- 不通过 noisy action 污染 observation bank；
- 不出现单 slot、单 camera 或单 coarse cell 的全局 sink；
- 地址不确定性在困难样本上可以保持较宽，而不是被强制单峰。

## 23. 一句话设计记忆

以后回看本设计时，应记住：

> 8×8 不是指向 84×84 固定 patch 的硬指针，而是拥有多个软地址 slot 的查询格；flow 提供可被语义纠正的几何 prior，G1/G2/G3 依次完成对齐、整流和坐标统一，W1/W2/W3 形成 horizon-conditioned posterior，P1/P2 最后读取高分辨率内容。Observation bank 可缓存，最终地址不可提前压缩，也不可由弱 flow 单独决定。

## 24. 能力边界与剩余问题

本节用于防止以后把 soft address lattice、3-3-2 和 AttnRes 当作同一个“万能修复”。三者分别处理：

- soft address lattice：高分辨率内容在什么位置、以多大不确定性被读取；
- 3-3-2 显式子角色：地址怎样经过对齐、整流、坐标统一、时间条件化和最终读取；
- AttnRes：不同深度产生的信息如何被选择、保留并送往下游。

### 24.1 能直接处理的问题

| 当前问题 | 对应机制 | 仍需通过的判据 |
|---|---|---|
| 84×84/42×42 候选过早压成每个 8×8 cell 一个 detail token | 保留多 slot、连续 fine candidates 和完整空间 candidate bank，推迟到 P1/P2 才读取 | joint address 优于旧 reader；detail intervention 能改变 action |
| 弱 learned flow 被迫单独充当空间地址 | flow 只作为 soft geometry prior，DINO、raw、G/W query 共同形成 posterior | flow-only、DINO-only、joint 三路消融 |
| 前三个 grounding block 只有同名角色，没有 G1/G2/G3 的现实职责 | clean address stream 明确承载 alignment、rectification、canonicalization 三类增量 | 三个 delta 的 zero/shuffle 对 posterior 影响不同且可解释 |
| world 已产生较大 anchor/camera residual，但其信息是否进入 policy 不清楚 | W1/W2/W3 形成 horizon/camera query；World Delta AttnRes 选择深度增量 | world query intervention、AttnRes 权重与 action delta |
| late detail attention 变尖但 update/trajectory ratio 持续缩小 | 受保护的 detail lane 和延迟读取避免高频信息在进入 policy 前被压缩或被主干吞没 | late detail update、梯度和 action intervention 同时不再接近零 |
| 当前流到 action 的因果效果接近零 | 新路径移除“先压缩、后选择”的结构性断点，并允许 action loss 自然回传到 joint address | 新版 zero/shuffle 必须在 action 或任务指标上产生显著差异 |

### 24.2 只能部分缓解的问题

| 当前问题 | 为什么只能部分缓解 | 还需要什么 |
|---|---|---|
| learned flow 本身趋近零流、moving/static 表现不对称 | joint address 可以容忍并纠正弱 flow，但不会自动把 flow predictor 训练好 | 单独检查 flow proposal；必要时改进双向匹配、occlusion/cycle/尺度监督和 uncertainty calibration |
| horizon 表征越来越相似 | horizon-conditioned address 和 World AttnRes 可保留不同空间证据，但无法补充数据中不存在的时间差异 | 记录 target adjacent/far cosine；做 horizon shuffle/zero；确认后再设计 change-conditioned temporal target |
| policy 梯度远大于 world/detail 梯度 | protected lane 和 Delta AttnRes 可改善传播，但梯度量还受模块规模、loss 和数据难度影响 | 同时报 update/action intervention，不能只按 grad norm 调权重 |
| gripper/event 时序和局部精度 | 高分辨率读取可能改善接触细节，却不能修复类别不平衡或 event/readout 口径错误 | event-aware 采样、decoded/event-head 闭环指标、必要时重做共享 event state/readout |
| 验证集早期平台 | 去掉结构瓶颈可能提高上限，但平台也可能来自单任务数据、动作平滑、采样和优化 | 完整 validation、episode 分层、precision-critical 子集和受控基线 |
| goal/action-history 利用率 | 它们可作为 address/world query 条件，但同一语言或单一历史分布无法证明条件有用 | goal/history zero、shuffle、truncate 探针以及多目标/多阶段数据 |

### 24.3 不在本结构的解决范围内

- 超出当前 observation window 的长期阶段记忆和任务进度估计；
- 数据集缺少多目标、阶段切换、失败恢复或精细操作样本；
- execution controller 的 capacity、hard route、value ranking 和执行消融；
- uncertainty NLL 的统计口径、方差上下界及其 loss 权重；
- action RMSE 与真实部署成功率之间的评价缺口；
- 日志版本标签、旧 checkpoint 继承和 preflight 文案；
- BF16 dtype 错误、OOM、吞吐和数据加载性能；
- bottom MMDiT、CVAE、workspace 或 execution 路径内部独立存在的问题。

这些问题不得通过提高 address 使用率、压低 entropy 或强制 flow 非零来“代偿”。

### 24.4 分轨治理顺序

1. 先补齐 V102 的完整 validation、target horizon similarity 和现有 late-detail/world/flow 因果探针。
2. 以 shadow mode 上线 observation bank 与 soft lattice，只观察 posterior、显存和几何正确性。
3. 只替换 late raw read，验证高分辨率 detail 对 action 是否真的有因果贡献。
4. 再接 G1/G2/G3 clean address stream，然后接 W1/W2/W3 horizon queries。
5. 地址路径被证明有效后，才引入 G→W、W→P 的 Delta AttnRes；policy→bottom MMDiT 属于后续独立阶段。
6. 其余问题分为五条独立实验轨：flow proposal、temporal/long-horizon、language/history、gripper/event、execution controller。

代码可以在同一条完整分支中实现，但实验必须按上述阶段分别启用，保留 V102 flags-off 等价路径。否则即使最终 RMSE 改善，也无法判断究竟是地址、深度选择还是别的旁路在起作用。
