# ClearVLA Schema38 采用记录：动作消费闭环与确定性代数修复

> 状态：已采用并进入 Schema38 实现。本文件保留采用时的设计依据、边界和
> 放行标准，不再是当前执行真相；实际源码契约、尚未验证的边界分别以
> `../00_CURRENT_ARCHITECTURE_CONTRACT.md` 和
> `../CURRENT_MAINLINE_ISSUES.md` 为准。不得从本计划推断测试或训练收益。

## 一、锚点与目标

下一版工作标签为 Schema38，capability 继续使用
`object_intent_dynamics_323`。三个锚点各自只承担一种职责：

- V120：乱改回归锚点。验证主路、bottom、训练/部署生命周期和已有能力
  没有再次丢失；它不是健康结构模板。
- V25：主要性能锚点。best physical RMSE 约 `0.07724`，epoch-8 约
  `0.07887`。
- V26：辅助性能锚点。best physical RMSE 约 `0.07893`。
- 健康度没有历史模板。Schema38 必须用可执行因果边界建立新的上界，
  不能把任一旧版本的公共化、旁路或尺度偶然性当作健康标准。

最新 Schema37 已完成七个验证 epoch，并运行到 epoch 8 batch 1660。
它没有数值崩溃，normalized RMSE、arm 和 horizon 已具有竞争力；主要性能
缺口集中在 gripper/精细动作。下一版不整体扩容或重写 top/bottom，而是
闭环修正 W/P2、dynamic P1/P3 precision 和 geometry 的动作消费，并处理
两个确定性代数/诊断错误。

## 二、最新日志对原方案的修正

1. W 时间公共化不是“W 完全没有读取”，而是 P2 强制消费 common、只让
   residual 经过 interval+null。公共未来是必选项，时间差异是可选修正。
2. public-S、typed-S、W 现在是三个异质的独立 interval 投票者。S 可在不
   理解 W 区间后果时直接选择时间；问题在 P2 consumer，不应再重写 S。
3. Static P1 仍健康。P3 precision 只有旧锚点约十分之一，是 Schema36/37
   将 dynamic P1 从 precision 完全切走造成的过度修正。
4. Geometry 没有坐标单位错配，禁止乘固定 cell-unit gain。确定错误是 W
   covariance 存在 Teacher 不可达到的正下界；结构问题是 geometry 只形成
   弱小值侧车，不能帮助选择高带宽 semantic successor。
5. G3 不应改成“先中心化再 tanh”。post-tanh common subtraction 对 K
   softmax 完全是 gauge；应做前向/梯度等价清理，而不是改变 Jacobian。
6. `gradient_global_preclip_l2` 是 20-batch 均值，owner 行却是当前 batch。
   Schema37 确有 observation 组有限尖峰，但当前日志不能定位参数；本轮只
   修诊断，不猜测性裁剪或改 observation 网络。

## 三、锁定的数据流

```text
G/S/CoarseAction
  -> W common + interval innovation
  -> reconstruct four complete interval fields
  -> action-conditioned K / KxC read inside each interval
  -> S conditions W keys, but cannot vote without W
  -> one type-local interval+null selection over complete W fields
  -> typed semantic/geometry consequence
  -> P3 effect/temporal

static V120 P1 + noisy action/time
  -> dynamic P1 policy-query residual
  -> P2 query                      (retained)
  -> fact-conditioned P3 precision (restored, zero-preserving)

P3 lanes + protected consequence
  -> unchanged Schema37/V120 bottom
```

W 对 future/effect/temporal 语义必须是必要载体，但整个 action 不得被强迫
经过 W。Static fact、precision 和 observable state-change 仍是合法的非 W
路径。

## 四、核心实现

### 1. P2：从 forced-common/optional-residual 改为完整 W field

W 内部仍可用 common/residual 表示和监督，但 P2 只能消费重构后的四个
完整字段：

```text
F[z,i] = W_common[z] + W_interval_residual[z,i]
z in semantic / geometry
i in 4-8 / 8-16 / 16-32 / 32-48
```

禁止存在第二个 common-only consumer。对每个 type、每个 interval，先完成
动作条件下的对象读取：

```text
semantic: K posterior
geometry: K x camera posterior

kW[z,i] = posterior-weighted key(F[z,i])
vW[z,i] = posterior-weighted value(F[z,i])
```

S 不再生成独立 interval logit。它只以零保持关系条件化 W key：

```text
cS[z,i] = bounded(public_S[i] + selected_typed_S[z,i])
conditioned_key[z,i] = kW[z,i] + kW[z,i] * tanh(cS[z,i])
logit[z,i] = similarity(action_query, conditioned_key[z,i])

posterior[z] = softmax(four interval logits + one type-local null)
effect[z] = sum_i posterior[z,i] * vW[z,i]
```

所有 key/value projection 必须 bias-free。由此得到：

- W 为零时 key/value/effect 精确为零，S 不能凭空造 future；
- S 为零时退化为普通 action-W 读取；
- common 与 residual 作为同一完整候选共同接受或拒绝；
- semantic/geometry 各有自己的 null，不竞争同一概率质量；
- null 只能拒绝 matching type 的 optional future effect；
- protected fact、precision、另一个 type 和 state-change 不受该 null 影响。

不加入 route quota、entropy target、hard gate、非空奖励或新外部 loss。

### 2. Dynamic P1：恢复合法 precision 消费，不恢复事实污染

现有 P2 query 保持：

```text
action_query + static_fact + policy_query_residual
```

P3 增加命名输入 `policy_query_residual`，但 dynamic P1 仍禁止写入
`factual_base`、protected consequence、transition 或 bottom protected base。

复用现有 projection 和 contract，不增加 block/loss：

```text
s = precision_innovation(static_fact)
d = smooth_rms_contract(policy_query_residual, 0.35)
g = tanh(variance_floored_centered_norm(s, 0.25))

precision_source = s + g * d
precision = precision_lane(tanh(precision_action(action_query)) * precision_source)
```

边界：

- `d=0` 时与 Schema37 static precision 数值/梯度等价；
- `s=0` 时 `g=0`，dynamic 不能只靠 action 凭空制造 precision；
- dynamic 只形成 fact-conditioned optional precision innovation；
- 大 dynamic RMS 不直接成为 protected value，也不靠任意 gain 恢复幅度。

### 3. Geometry：既保留独立 value，又参与 semantic 空间地址

现有 geometry KxC posterior 不再只是低维 transport value 的选择器。将其
在 camera 轴上的 K 边缘证据做 valid-K 中心化和有界化，并只在同一
interval 内校正 semantic K logits：

```text
geometry_k_evidence = bounded_centered(
    logsumexp_camera(geometry_kc_logits + log(camera_support))
)

semantic_k_logits = semantic_k_logits + geometry_k_evidence
```

- 无有效 camera/geometry 时 correction 精确为零，semantic read 数值等价；
- 均匀 geometry 证据中心化后为零；
- geometry 只能改变 interval 内 K 地址，不能直接投票 interval；
- semantic value 不被 geometry 缩放、压缩或替换；
- geometry 独立 effect lane 继续保留；
- action loss 可通过高带宽 semantic successor 普通反传到 geometry spatial
  evidence；不得 detach 或注入人工梯度。

同时修复 covariance 可达性。Teacher covariance 可为零，W 输出改为无正
下界的 bounded PSD 参数化；保持当前约 `0.125` 的初始化方差：

```text
variance_xx = sigmoid(raw_xx)
variance_yy = sigmoid(raw_yy)
xy = tanh(raw_corr) * sqrt(variance_xx * variance_yy)
```

bias 调整为使初始方差与 Schema37 一致。P2 的 `1/49` 仅作为 Mahalanobis
metric 数值 floor 保留，不写回 W target/value。Transport、Teacher、flow 和
坐标仍使用现有 normalized-image units。

### 4. G3：只做数学等价的 gauge 清理

当前：

```text
raw = 0.5*tanh(head)
common = E_parent(raw)
corrected = softmax(log(parent) + raw - common)
```

改为：

```text
raw = 0.5*tanh(head)
corrected = softmax(log(parent) + raw)
```

同一 scalar common 对 K softmax 精确不变，因此这是函数与 Jacobian 的等价
清理，不是能力修改。保留 K-vs-null mass。删除误导性的 centered residual
RMS 和近零 margin ratio，改记：raw RMS/max、pairwise logit span、posterior
L1 和 assignment-change fraction。

禁止改成 pre-tanh centering；那会改变函数与梯度，超出本轮边界。

### 5. 梯度与路由诊断

不改变 optimizer、clip、scheduler 或 step：

- `gradient_global_preclip_l2` 改名
  `gradient_window_preclip_l2_mean`；
- 同时记录 window max、对应 batch offset/global step 和当前 batch norm；
- global norm 超过 audit threshold 时，在任何 clip 前写独立
  `gradient_spike` 行：最大 L2 参数、最大绝对值参数、完整参数名、role、
  optimizer group、shape/dtype、epoch/batch/step；
- 正常 batch 不逐参数扫描，避免吞吐回归；
- 该 audit 开关前后参数更新必须数值等价。

P2 只在 diagnostic batch 增加：

- public-S、typed-S、W key 的 interval-common offset RMS；
- interval-centered variation；
- pre-tanh derivative/saturation fraction；
- S-condition neutral、geometry-condition neutral 的 posterior L1；
- full-field type-local null/non-null mass。

清理重复/误导别名，尤其不再把 `p1_completed_fact_rms` 称为事实 RMS。

## 五、明确不改

- 不改 Teacher OT/dustbin/reliability/target 代数；
- 不改 learned-flow gain、loss 或强迫非零流；
- 不改 S block、S target、语言/历史输入或增加阶段监督；
- 不改 W1/W2 block 数、外部 future loss 权重或加入多样性损失；
- 不重建 static V120 P1；
- 不改 controlled transition、Evidence MMDiT、CVAE/workspace、execution、
  action/event/motion heads 和 bottom lane-local routing；
- 不增加 gain、quota、hard gate、entropy target、人工梯度或新辅助 loss。

## 六、代码组织、版本与训练身份

- manifest 升为 Schema38；capability 名称不变。
- Schema37 exact/optimizer resume 拒绝；正式实验 fresh run。
- bottom ABI 不变；显式 bottom-only migration 工具可保留，但正式比较不用。
- 设计不要求新增参数；covariance 只改参数化/bias，P2/P3 复用现有模块。
- 外部 loss 和 optimizer group 不变。
- 当前架构契约只在实现、测试和三轮审查完成后更新；本文件在此前始终是
  auxiliary prospective plan。

建议在一个 Schema38 中完整整合，但按 owner 分成可审查提交：

1. 等价清理与诊断：G3 gauge、gradient window/spike、metric aliases；
2. W full-field + P2 S-conditioned interval read；
3. dynamic P1 -> P3 precision；
4. geometry semantic-address condition + covariance reachability；
5. manifest、测试、日志、文档。

不为每个提交各跑一次长实验；通过独立接口测试和 causal diagnostics 归因，
最终只放行一个完整 Schema38 实验。

## 七、测试与三轮审查

### 结构/代数测试

- `F_i == common + residual_i`；源码中无第二个 common consumer。
- W neutral 时 P2 typed effect 精确零，consequence 回到 static fact，四条
  P3 effect/temporal 精确零；precision/state-change 保持合法。
- W neutral 时任意 S intervention 不能改变 future effect。
- S neutral 时 action-W full-field read 仍可工作。
- type-local null 不能改变另一个 type、static fact、precision 或 protected
  consequence。
- dynamic P1 为零时 static precision 数值/梯度等价；static fact 为零时
  dynamic precision 精确为零。
- dynamic intervention 同时可改变 P2 query 和 P3 precision，但不能改变
  protected fact/transition factual source。
- geometry unavailable/uniform 时 semantic posterior/output 数值等价。
- geometry intervention 只改变同 interval 的 K/KxC selection，不直接改变
  interval prior；K/camera permutation 等变。
- covariance 初始输出保持、零 target 可渐近达到、矩阵始终 PSD、BF16 有限。
- G3 新旧 posterior 与参数梯度 allclose，raw max <= 0.5，K/null mass 不变。

### 反向所有权测试

只用最终 action loss，JVP/VJP 必须分别到达：

- W complete common/interval field；
- W interval innovation；
- S-conditioned W key；
- dynamic P1 precision interaction；
- semantic value；
- geometry spatial evidence、transport 和 covariance。

不得以“参数有梯度”代替边界张量的因果变化。

### 生命周期/兼容测试

- Teacher：每训练 batch 一次，deploy 零次；
- observation/G/S/W/static P1：每 observation 一次；
- dynamic P1/P2/P3/bottom：五个 Euler 节点加 endpoint head；
- endpoint forward 不改变最终 action；
- bottom 输入相同时输出数值等价；
- 全空 camera/K、全 null、全 neutral W、BF16 forward/backward 均有限；
- 所有参数有唯一 optimizer owner；Schema37 resume 明确拒绝。

三轮审查：

1. producer -> consumer：逐轴检查 K/type/camera/interval/lane、zero/null、重复
   carrier 和旁路；
2. consumer -> producer：从 action loss 反查普通梯度、scale/Jacobian、loss
   owner、detach/no-grad 和被压缩方向；
3. runtime/checkpoint：调用频率、Teacher isolation、ODE 静态缓存、参数/RNG、
   BF16、显存、吞吐、fresh/resume 身份。

## 八、实验放行

### Smoke

先通过 BF16 train forward/backward、五步 deploy、Teacher isolation、neutral
W、dynamic precision、geometry/K/camera permutation、G3 equivalence、gradient
spike logger 和 resume rejection。

### Epoch-1 安全 gate

这是防灾 gate，不是性能结论：

- physical RMSE 不得比 Schema37 epoch 1 (`0.10287`) 恶化超过 5%；
- first/tail、三个 horizon、arm/gripper、event/motion 任一不得出现超过 10%
  的无补偿退化；
- W full-field、dynamic precision、geometry semantic address 的 action JVP
  均必须可定位；
- 不得出现 hidden non-finite；有限 spike 若出现必须记录精确参数；
- batch 8 总显存低于 22 GiB，目标不高于 Schema37 约 `11.9 GiB + 1.5 GiB`；
- median throughput 不慢于约 `2.2 s/batch`。

### 八轮评价

- V120：检查主路/lifecycle 无回归，final physical 不劣于其约 `0.08145`；
- V25：主要性能目标，比较 best `0.07724`、epoch-8 `0.07887` 和八轮均值；
- V26：辅助性能比较，best 约 `0.07893`；
- 不只看 aggregate：arm、gripper、first/tail、三 horizon、event、motion、
  execution 和 epoch 6-8 rebound 必须同时比较；
- 不给 W variation、G3 correction、flow、null mass 或 lane RMS 设训练目标；
  健康度以无旁路、零语义、轴保留和 causal JVP 判断；
- 若结构边界全部闭环而 W 区间仍不具动作收益，则归入 Teacher/data
  identifiability，而不是继续加 gain、quota 或辅助 loss。

## 九、完成定义

Schema38 只有同时满足以下条件才算完成：

1. W future effect 不能由 S 独立伪造，common/residual 不再被拆成 mandatory
   与 optional 两条消费者；
2. dynamic P1 恢复合法 precision 带宽，但不能成为第二个 protected fact；
3. geometry 同时拥有独立 value 和 semantic-address 作用，且 covariance target
   可达；
4. G3/gradient 指标不再用错误代数掩盖真实状态；
5. V120 回归完整、V25/V26 性能可比较、结构健康度高于所有历史版本。
