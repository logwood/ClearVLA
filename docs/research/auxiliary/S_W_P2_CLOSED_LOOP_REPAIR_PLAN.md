# S–W–P2 闭环根治计划

状态：已按 Schema31 实现；当前语义已提升到
`../00_CURRENT_ARCHITECTURE_CONTRACT.md`，本文仅保留实施依据与边界。

本计划只处理 `CURRENT_MAINLINE_ISSUES.md` 中 O-02～O-06 形成的连续问题：

```text
S common 先行
  + Teacher 低辨识软平均
  -> W 最终 field 公共化
  -> P2 用全局 null 廉价少读 W
```

“最小”只表示不重建 G、P1、transition、bottom，也不增加新的 DiT block；它不允许只改一个
局部公式。只有 `Teacher -> S -> W -> P2 -> consequence -> action` 同时闭合，才算完成。

## 一、日志与源码依据

- Schema30 epoch 3：S semantic common/differential score 为 `0.752/0.382`，geometry 为
  `0.863/0.230`；condition-centered interval variation 仅为 semantic `0.0275`、geometry
  `0.0113`、appearance `0.0147`。
- Teacher association confidence/reliability 为 `0.291/0.289`，null 仅 `0.0458`，semantic
  target RMS 却为 `0.4586`。低辨识空间平均仍成为完整目标。
- W prediction interval variation 为 `0.0495`，Teacher 为 `0.1019`；W hidden 有明显变化，
  但最终 field 只保留约一半。
- P2 posterior max 与 null mass 同为 `0.6930`；semantic/geometry/status null 分别为
  `0.7575/0.6912/0.6305`。null 基本就是 top-1，而非候选数被低估。
- 当前源码把 S common/differential score 先算术平均；W 的 semantic/geometry 候选支持又受
  W 自己预测的 future visibility 影响；P2 的一个 null 可以把整个对应类型的 W value 归零。

## 二、锁定的闭环结构

```text
current facts + goal/history
    -> S common object/type relevance          [B,K,type,R]
    -> S signed interval residual              [B,I,K,type,R]

future supports
    -> no-grad partial matching + dustbin
    -> common future effect                    [B,K,*]
    -> zero-mean interval residual             [B,I,K,*]

S common/residual + current facts + clean coarse action
    -> W common effect + W interval residual
    -> reconstructed FutureObjectDynamics

P1 factual base
    + P2 protected common-effect read          # 没有 learned null
    + P2 optional interval-residual read       # 保留显式零值 null
    -> zero-preserving consequence
    -> unchanged P3 / transition / V120 bottom
```

闭环原则：

1. common 只能解释跨区间稳定的未来后果，不能伪装成 interval difference；
2. interval residual 必须是 signed、zero-centred、condition-owned 的真实值；
3. W loss 与 P2 消费的是同一个 common/residual field，不允许第二个 hidden carrier；
4. null 只能拒绝可选 residual，不能一次抹掉 W 的全部受监督后果；
5. neutral W 仍严格映射到零 effect，不能通过 bias、uncertainty 或 validity 制造默认值。

## 三、Teacher：从单边 softmax 改为带 dustbin 的软部分匹配

保留现有 full-DINO value、semantic/appearance key、camera prior、flow geometry 和 FP32/no-grad
边界；只替换关联代数。

### 1. 关联

- 每个 future support 构造 `[K, C*8*8]` score matrix。
- semantic、appearance 分别减去各自的空间背景基线，再与相对 geometry/camera score 相加。
  广泛的正 cosine 因而不能只靠候选数量压低 null。
- 使用小矩阵、log-space partial optimal transport：真实 K 行、真实未来 cell 列，并加入
  dustbin row/column。dustbin score 固定，不可学习；Teacher 也不接收 action loss。
- 保持软对应，不使用 argmax、hard crop、固定熵目标或匹配配额。
- 输出真实候选 posterior、dustbin mass、effective support、mutual-match mass、
  best-minus-background 和空间半径。

这是从 SuperGlue 的 partial assignment/dustbin 边界借用“显式不可匹配”语义；矩阵只有
`(K+1) x (C*64+1)`，每 batch 12 次、且 no-grad，不需要新增在线网络或显著显存。

### 2. 目标分解

先按四个原区间均匀聚合 support，得到完整的 per-interval effect：

```text
full_effect_i = successor_i - current_reference
common_effect = mean_i(full_effect_i)
interval_residual_i = full_effect_i - common_effect
```

因此严格满足：

```text
full_effect_i == common_effect + interval_residual_i
mean_i(interval_residual_i) == 0
```

semantic、transport、visibility/persistence 均按该方式保留 common/residual；covariance 与
uncertainty 保持校准字段，不进入 P2 value。dustbin 产生 identity、零 transport/status change 和高
uncertainty，而不是删除该 loss row。reliability 继续只做诊断/校准，不作为 loss mask。

## 四、S：common 是 base，differential 是 value，永不提前相加

替换当前：

```text
score = 0.5 * (common_score + differential_score)
```

为两个显式边界：

```text
typed_common_value       [B,K,type,R]
typed_interval_residual  [B,I,K,type,R]
```

- 两者复用现有 `typed_relevance_queries` 和 typed fact projections，不增加 S block。
- common query 读取跨区间 observable carrier，只预测 Teacher common effect。
- differential query 只读 `interval - interval_mean`，只预测 Teacher interval residual。
- 两条路径均保持 variance floor、bias-free、signed zero-null；四区间输入相同则 residual
  必须 bit-exact 为零。
- `WorldIntentDock` 分别暴露 common 和 residual；不再暴露预混合的
  `typed_relevance_value`。
- `CoarseActionIntent` 删除 typed value reader/router。它仍读取 public condition、history 和
  public-free object content，并继续用同一个 online tensor接受 clean future-action 辅助监督；typed
  K/type 只通过 `WorldIntentDock` 进入 W 一次。
- P1 只允许使用 common/goal 作为地址 query 条件，不读取 interval residual value；P1 value 仍只来
  自当前 RGB/detail。
- P3 temporal 保留现状，因为它必须乘 protected consequence；不新增 S 到 bottom 的独立 value。

## 五、W：分别预测 common effect 和 interval residual

不增加 W block，继续使用 W1=`4–8/8–16`、W2=`16–32/32–48`。

- common typed state 由 S common value、当前 object facts、goal 与 clean coarse action 构造；它在
  interval softmax 外保留一次，并使用现有 typed projections/field heads解码 common effect。
- interval typed state 只由 S interval residual value 初始化，并穿过现有 W1/W2 typed blocks。
- public W hidden 只能用有界乘法调制已经非零的 typed state，不能加法制造 common 或 residual。
- field heads 输出 `common + residual` 的同一 `FutureObjectDynamics`；P2 不得读取另一个 W hidden。
- Direct S supervision 使用完全相同的 field heads：common 对 common target，residual 对 residual
  target。它不再用一份混合 field 同时声明三种所有权。

这样 W 内部的 interval variation 只有两种合法去向：成为被监督 residual，或者在 W block 中被证明
丢失；不会再被 common carrier 解释掉。

## 六、P2：protected common read + optional residual read

### 1. common read

- 对每个 semantic/geometry/status 类型，仅在 K 个当前物理有效 object 上做条件读；没有 learned
  null。
- 没有物理有效 object 时，通过支持分母显式返回零；common effect 本身为零时读出也严格为零。
- common object selection 可读 action query、S common key 与 W common source key，但不读取
  predicted uncertainty/reliability。

### 2. interval residual read

- 保留当前每类型独立的 interval×K posterior 和零值 null。
- null 只控制 residual，不得影响 protected common read。
- semantic、geometry、status 的候选支持全部来自 detached current physical validity/existence；
  W 自己预测的 visibility/persistence 不得再掩掉 semantic/geometry 候选。
- object-common score 与 interval-residual score 作为不同轴的 logit 因子相加，不先合成一个
  vector 或一个 tanh gate。
- 三类型继续使用当前 `sum/sqrt(3)` protected fusion 与 contrast-only residual；不恢复 outer type
  softmax。

最终：

```text
effect = common_read + residual_read
interaction = bias_free(tanh(P1_fact) * effect)
protected_consequence = P1_fact + effect + interaction
```

因此 residual null 可以表示“当前动作不需要区间微调”，但不能表示“忽略整个 W”。这比负 null
bias、mass quota 或 hard gate 更接近真正的结构根治。

## 七、loss 重写但不增加外部预算

- `future_dynamics=0.10`、`intent_structure=0.02` 总预算不增加。
- 删除代数重复的 `successor` 与 `semantic_delta` 双重 raw 监督；用 common-effect 与
  interval-residual 两项替代，二者之和严格重构原 full effect。
- 原 semantic 总份额保持不增加；common/residual 在内部做固定、可审计的分配。
- semantic 用 current-reference 尺度的 detached fixed-floor chart；transport 使用 normalized
  coordinate units；status 使用 `[-1,1]` 单位。三者不再直接做未校准算术平均。
- 不按预测 uncertainty、reliability、null mass 或 selector mass缩 loss；这些值无法成为
  self-erasing shortcut。
- 原 adjacent transition 项改为只监督 interval residual 的相邻差，避免 common 被重复计算。

## 八、明确保持不变

- G1/G2/G3、global-K binder 和当前 grounder loss；
- exact V120 P1：24 queries、N=49、3×3 microgrid；
- P3 四 lane 的来源语义；
- controlled transition、CVAE、workspace、Evidence MMDiT、capacity、execution；
- flow 网络与几何 loss；
- flow-time、五步 Euler、endpoint heads、optimizer/clip 生命周期；
- 单阶段端到端训练与所有外部 objective 权重。

接口变化需要新的 manifest schema 和 fresh run；旧 top checkpoint 不允许 exact resume。bottom-only
migration 仍必须显式报告，但正式对照优先 fresh。

## 九、实施顺序

1. 先扩展 typed interfaces 与纯代数测试，不接主训练。
2. 实现 Teacher partial assignment 与 common/residual target pack；验证 Teacher-only 替换不会改变
   deployment action。
3. 拆 S common/residual，移除 CoarseAction typed duplicate；保持参数初始化顺序可解释。
4. W 输出同一 common/residual field，并让 direct supervisor 与 P2 消费同一对象。
5. P2 拆 protected common read 和 optional residual read，关闭 predicted visibility self-mask。
6. 重写现有预算内 loss 与日志；不添加新 objective。
7. 三轮审查：provenance/旁路；dtype/zero/Jacobian/optimizer；cache/Teacher/吞吐/显存。

## 十、闭环验收

### 结构验收

- 四个相同 interval 输入时，S/W interval residual bit-exact 为零，common 保留。
- 只扰动一个 interval 时，仅相应 residual 与依赖它的远端状态改变；common 改变量符合均值代数。
- Teacher target 满足 full=`common+residual`、residual interval mean=`0`。
- diffuse 相同分数不能生成任意方向的 semantic/transport target；必须产生 dustbin/identity 与高
  uncertainty。尖锐软匹配仍能恢复非等像素 transport。
- common S 路径不能直接改变 interval residual；residual 路径不能改变 common field。
- CoarseAction 源码和 interface 中不存在 typed K/type value reader。
- 非零 public W、goal、action 在 typed common/residual 全零时不能制造 FutureObjectDynamics value。
- predicted visibility 改变不得改变 semantic/geometry candidate support。
- common effect 非零时 residual null 不能把完整 P2 effect 变成零；common/residual 全零时 effect 与
  interaction 必须 exact zero。
- Teacher 构建训练每 batch 一次、部署零次；G/S/W/P1 仍每 observation 一次。

### 日志验收

- 分别记录 S common/residual score、value、condition-centred variation 与 JVP；
- Teacher 记录 dustbin、effective support、mutual mass、background margin、interval residual RMS；
- W 分别记录 common 与 residual 的 prediction/target error、decoder 前后 variation；
- P2 分别记录 protected common RMS、residual RMS、residual null、三类型贡献；
- 原 `object_p2_null_mass` 重命名为 residual-null，避免再把它解释为整个 W 的采用率。

### 动作闭环验收

在同一冻结 checkpoint 上依次做：

```text
S common zero/shuffle
S interval residual zero/shuffle
W common effect zero/shuffle
W interval residual zero/shuffle
P2 residual-null forced zero/one（仅 audit）
```

每项都按 `source boundary -> W field -> P2 read -> consequence -> action` 报告效应与置信区间。
只有 W common/residual 的边界和 action 均改变，且 zeroing W 不再系统性改善误差，才宣称闭环成立。
不设置 null、entropy、variation 或 JVP 的目标数值。

## 十一、第二遍旁路统合检查

- S common 不再进入 residual value，但可作为 W/P2 的 key；key 不得经 bias 投影制造 value。
- CoarseAction 仍可作为 W 的 clean action condition，但不再重复携带 typed value。
- P1 可被 goal/common 调整地址，但其 value 只能是当前事实，不能复制 future effect。
- P3 temporal 仍可承载 stateless control，但必须乘 protected consequence；没有第二个 S future value
  直达 bottom。
- W predicted visibility 仍是有意义的 status value，但不再拥有抹除 semantic/geometry 的权力。
- residual null 仍合法，但它只拥有 optional delta，不能拥有 protected common effect。
- bottom learned constant、P1 self-write、G3 corrector 与 learned flow 是独立账目，不在本轮顺手
  修改，也不能被本轮成功指标自动宣告解决。
