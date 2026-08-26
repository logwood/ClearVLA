# P2 分层区间—对象软路由备选修正

状态：**Schema32 时期的条件性局部方案，未作为当前 replay 决策采用。** 原始触发
条件仅作为历史设计依据；是否在 Schema25 重放中恢复其不变量，由
`ARCHITECTURE_REPLAY_PLAN.md` 的逐修改审查决定。代码与 ABI 范围小，但训练和
动作行为影响为中等，不能作为无感 hotfix 或旧 checkpoint 的续训补丁。

本文件描述 P2 的核心代数，不规定未来必须“一个微小改动跑一个实验”。ClearVLA 的
修复单元应是一个职责闭合的模块或连续数据流边界。如果 Schema32 证明 S interval key、
W residual identifiability 与 P2 factorization 属于同一个根因链，可在同一模块版本中组合
修正；但每项仍需独立的边界指标、代码所有权和可撤销提交。Teacher dustbin 只有在证据
证明它直接限制同一 residual target 时才进入该组合，不能仅因同时异常而捆绑。

原方案要求 Schema32 的对齐日志证明 W 输出健康、但区间残差继续在 W→P2 边界
显著衰减后才进入实现。该条件现在只解释方案的历史证据边界，不自动要求补跑实验，
也不允许越过 replay register 直接进入实现。

## 1. 触发证据

V31 的同一窗口内：

```text
Teacher residual/common = 0.0760 / 0.1713 = 0.444
W       residual/common = 0.0464 / 0.1101 = 0.421
P2      residual/common = 0.0259 / 0.1719 = 0.151

relative retention = 0.151 / 0.421 = 0.36
residual posterior max = 0.224
residual posterior entropy = 0.788
residual null mass = 0.027
```

这说明当时主要不是 null 丢弃，而是 P2 在 `interval × object` 联合支持上做一次
softmax 后，对 signed、跨区间零均值 residual 做凸组合；分散 posterior 会让不同区间的
正负变化提前相消。

只有同时满足下列条件才启动本修正：

1. Schema32 的 Teacher→W residual/common 相对比例保持健康；
2. W1/W2 common/residual processing 与 typed-by-base interaction 已真实学习，不是近零空转；
3. P2/W 相对残差保留率没有明显关闭 V31 `0.36 → 1.0` 的差距；
4. residual null mass 不高，排除 null shortcut；
5. P2 cancellation 指标与 tail/far-horizon 停滞方向一致。

以上证据不成立时，不修改 P2。

## 2. 修改边界

本修正只改 `ObjectFutureEffectReader` 的 interval-residual 选择代数及配套 typed interface、
日志和测试：

- G、Teacher、S、W1/W2、P1、P3、transition 和 bottom 不改；
- common-effect K read 不改，始终位于 residual optional read 之外；
- semantic、geometry、status 仍是三个互补 value，不恢复 outer type softmax；
- 不增加 loss、gain、quota、熵目标、硬门控、阶段标签或 progress 条件；
- 不使用 predicted future visibility 作为路由 authority；
- 不重复读取 RGB/DINO，不改变 Teacher 生命周期；
- 默认 fresh run，新代数不得 exact-resume 旧 top checkpoint。

### 2.1 影响分级

```text
代码范围：       小（P2 reader、typed diagnostics/tests、manifest）
张量/API 范围： 小（P2 输出 shape 与 consequence/P3/bottom ABI 不变）
在线行为影响：   中（每个 ODE 动态 forward 的 effect 方向与区间采用会变化）
训练几何影响：   中（action loss 会重新分配到 S interval key 与 W residual field）
下游尺度风险：   中低（现有 P2 0.35 RMS contract 与 P3 lane contract 继续生效）
checkpoint：     新 schema、fresh run；旧 top 不允许 exact resume
```

具体引申边界：

- `effect -> consequence -> P3 effect/temporal -> bottom` 的数值方向会变化，这是修正
  的目的，不能宣称 bit-exact；
- consequence 的 algebraic zero、P3/Pbottom shape 和现有 RMS 上界保持不变，因此不需要
  改下游模块；
- W/Teacher loss 不变，但 action gradient 对 S/W 的分配会变化，必须记录 owner-level raw
  gradient 并与 Schema32 同 iter 比较；
- 如果实现需要改 S/W 输出 shape、增加监督或调整 P3/bottom scale，即视为超出本计划，
  停止并重新审查，不得以“P2 小修”名义扩散。

## 3. 目标代数

对每个 action horizon query `q` 和每个类型 `z`，把当前联合选择：

```text
p(i,k,null | q,z) = softmax over (interval × object + null)
```

改成职责分离的软分解：

```text
pi(i or null | q,z) = temporal softmax
rho(k | i,q,z)      = object softmax inside interval i

read(i,q,z) = sum_k rho(k | i,q,z) * residual(i,k,z)
read(q,z)   = sum_i pi(i | q,z) * read(i,q,z)
```

等价记号：

```text
p(i,k | q,z) = pi(i | q,z) * rho(k | i,q,z)
```

但两个 posterior 的证据所有权不同，不能重新退化成对原 joint logits 的精确
`logsumexp` 分解：

### 3.1 区间 posterior

`pi` 只决定“这个动作位置应读取哪个未来区间”：

```text
interval_logit(i)
    = tau_intent * cosine(
          action-time query,
          S public interval-residual key(i)
      )
```

- action query 已包含真实 horizon/time identity；
- S interval key 是在线 observable intent，不读 frame progress 或 Teacher；
- 不把 K 数量、某个对象的强 score 或 geometry 重复计入区间 posterior；
- 使用一个全局 residual null，value 精确为零；null 不复制四次；
- 没有任何有效 K support 的 interval 被构造性 mask；全无支持时输出精确零。

第一版复用现有 bounded intent temperature，不新增自由增益。

### 3.2 区间内对象 posterior

`rho` 只决定“在已经给定的区间内，哪个对象的后果与当前动作相关”：

```text
object_logit(i,k)
    = tau_content * cosine(action query, W field key(i,k))
    + tau_intent  * cosine(action intent query, S typed residual key(i,k))
    + tau_coord   * bounded real-camera-mixture score(i,k)  # geometry only
    + log(current physical K support)
```

- 每个 interval 内只对物理 K 做 softmax；
- inner object read 不另设 learned null，零作用由 outer residual null 表达；
- geometry 继续使用 Schema32 的真实 camera mixture；
- invalid K 精确为零支持；K permutation 保持等变。

### 3.3 Value 与融合

- residual value 仍直接来自被监督的 `FutureObjectDynamics`；
- 四个 per-interval read 在 `pi` 之后才组合，禁止在 temporal posterior 建立前求平均；
- signed residual 在真正的 temporal uncertainty 下仍允许相消，这属于合法期望，而不是错误；
- common value 与 residual value 最后相加，common 不受 residual null 影响；
- semantic/geometry/status 继续使用现有 `sum/sqrt(3)` protected fusion base 与
  contrast-only residual；不改变 P3 consequence 接口。

## 4. 为什么这是局部修正而不是新架构

- 不新增 block 或监督目标；
- 复用现有 action/source/intent queries、keys、values 和 bounded temperatures；
- 计算规模由一次至多 17 项 softmax，变成四项 interval softmax加四个至多 4 项 object
  softmax，额外开销相对完整模型可忽略；
- 输出 shape、P2→consequence→P3 ABI 不变；
- 改变的只是“先选择时间，再选择对象”的归一化域和 null 所有权。

## 5. 必须新增的诊断

不把以下指标加入 loss：

```text
P2 interval posterior entropy/max/null
P2 interval mass[4] and action-horizon variation
P2 within-interval object entropy/max/effective-count
P2 residual cancellation ratio
P2 residual/common ratio
P2-to-W relative residual retention
per-type semantic/geometry/status residual retention
```

其中 cancellation ratio 定义为：

```text
norm(sum_i pi_i * read_i)
---------------------------------
sum_i pi_i * norm(read_i) + eps
```

它只说明跨区间组合发生了多少方向抵消，不规定越大越好。

## 6. 可执行测试

1. K permutation：输入 K 重排只重排 inner posterior，最终 read 等变；
2. interval isolation：改变 interval logits 不改变各 interval 内的 object posterior；
3. object isolation：改变某 interval 内 object logits 不改变 outer interval posterior；
4. one-hot temporal：选择单一区间时，其他区间 residual 不得影响输出；
5. global null：null=1 时 residual 精确为零，protected common bit-exact；
6. zero semantics：全部 residual 为零时 P2 residual、interaction 和对应 consequence 增量精确为零；
7. support：部分/全部 invalid K 时有限且无虚假 geometry evidence；
8. camera permutation：geometry inner posterior 与 read 对相机排列不变；
9. complementary type：任一 type zero/shuffle 只改变自身 value，不能淘汰另外两类；
10. Teacher isolation：替换未来 supports 只能改变 target/loss，部署 action bit-exact；
11. BF16 forward/backward、普通 autograd、optimizer ownership 与五步部署通过；
12. 旧 joint 与新 factorized 路由使用相同冻结输入做边界对比，但不要求数值 bit-exact。

## 7. 实验放行与回滚

先做相同 seed/data/batch 的 fresh smoke，再在相同 iter 比较 Schema32：

- Teacher→W 比例不得恶化；
- P2-to-W residual retention 必须明显提高，且不能依靠 common 人为缩小、residual RMS
  爆大或 null mass 受压实现；
- P2 interval posterior 必须随 action horizon/S intent 改变，不能固定到一个区间；
- arm/tail/far horizon 不得退化，gripper/event 改善不得以精细动作带宽为代价；
- 完整八轮检查后期反弹。

若边界保留率改善但 action 无收益，归类为数据/可识别性问题并回滚该代数，不再向 P2
叠加 gain 或辅助 loss。若 Schema32 已自然恢复 P2 residual 带宽，则本方案不实施。

这里的“归因与回滚”按模块闭环理解：

- 一个实验可以同时包含同根因链上的 S key、W residual 与 P2 routing 配套修正；
- 用同一运行中的 `S boundary -> W field -> P2 read -> consequence -> action` 指标分层归因，
  不要求每个子改动各跑八轮；
- 子改动分提交、分 owner、分诊断，以便静态复核和必要时部分撤销；
- 单个子改动不必独立提高 aggregate RMSE。若它关闭了确定结构缺陷、没有引入退化且是
  后续闭环的必要条件，可以保留；最终收益由完整模块和全程曲线判断；
- 不把无共同数据流根因的 G3、P1 self-write、flow 或 bottom 风险塞进同一个修复包。
