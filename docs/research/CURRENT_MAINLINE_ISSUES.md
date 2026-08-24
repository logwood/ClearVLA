# ClearVLA 当前主线纯问题账本

当前源码身份：Schema36 `object_intent_dynamics_323`，工作树基于提交
`03235d313f96c7782e79e24fcb8e4fdc9e9543d8`。行为锚点仍是 V120 `long` 与本地完整快照
`.audit/v120_exact_source_0b92d359/`；Schema35 只用于识别本轮修复的 P1/P2 回归，不是正确性目标。

本文件只记录当前仍未关闭的问题。已经由源码代数和测试关闭的 G 重复计票、Teacher
dustbin/status 混淆、camera 几何压平、S 绝对未来均值、W far->near、P3 temporal 事实旁路和
Teacher leak 不再保留为历史条目。

## 全链路审查门

以后不得再从单个异常指标向两侧做局部补洞。修改前后都必须完成三张相互核对的图：

1. 正向在线图：数据/缓存 -> Observation/Pre-G -> G1/G2/G3 -> S/W -> static P1 ->
   dynamic P1/P2/P3 -> controlled transition -> V120 bottom -> action/event/motion。
2. 训练图：future supports -> no-grad Teacher -> 每一项 loss -> 实际被监督张量；同时核对
   Teacher 隔离、每 batch/每 ODE 调用次数和部署零调用。
3. 反向图：每一项 loss -> 每个参数 owner -> optimizer group -> local/global clip -> checkpoint
   与日志。必须核对轴、zero semantics、dtype、幅度、Jacobian、旁路和重复消费。

只有 shape 正确、张量非空、存在梯度、单测通过或接口有名字，都不能关闭问题。一个条目只有在
producer、全部 transformation、全部 consumer、loss、反向 owner 和部署生命周期都解释完整后
才能删除。

## O-01：dynamic P1 已发生确定的饱和与隐藏 Jacobian 爆炸

**级别：P0。类型：确定性训练故障。**

Schema35 在 epoch 3 的 `batch 1080 -> 1160` 期间，forward loss 和所有公开张量仍有限，但 raw
gradient 出现如下升级：

```text
batch             1080      1100      1120      1140       1160
global L2         1.31      3.12      5.10      1.02e3     5.95e6
dynamic P1 L2     0.277     1.84      4.65      8.32e2     5.17e6
canvas seed L2    0.545     2.32      1.87      5.88e2     2.94e6
```

`batch 1170` 首个 non-finite 参数是
`observation.encoder.flow.correlation_temperature_log`，但它位于已经被 P1 巨大反向增益穿过的
上游 Observation 路径，只能视为首个受害参数，不能视为根因。

同时，Schema34 同一位置的 dynamic P1 约为 `0.275`，self/FFN write 约为
`0.137/0.219`；Schema35 dynamic P1 约为 `0.81`，self/FFN write 已顶到各自约 `0.5` 的接口
上限，P3 precision 也顶到约 `0.347/0.35`。Schema35 新增的
`static_precision + dynamic_precision` 直路使 action loss 可以把 dynamic P1 当成高容量 precision
adapter；这与新回归时间和饱和方向一致。

当前 P1 block 只约束 forward residual RMS。其 AdaLN shift/scale、attention Q/K 和
`p1_content_mod_scale` 的完整 Jacobian 没有同等边界；输出有限不代表反向有限。现有日志还丢弃了
该 block 已经生成的 raw/proposed/compression、normalization denominator/gain 指标，因此尚不能在
self-attention、FFN 和 modulation 三者间确定第一个内部放大点。

**关闭边界：**先恢复完整 P1 内部数值诊断并在相同数据顺序上复现爆炸前窗口；必须定位第一个
finite-but-growing 内部量和对应参数。修复要同时保证 dynamic P1 不再长期贴住接口上限、隐藏
Jacobian 有构造边界、正常小信号不被粗暴缩小。仅增加局部裁剪或缩小 gain 不能关闭本条。

Schema36 已完成源码侧修复：dynamic P1 不再拥有 P3 precision 的第二出口；policy AdaLN
shift/scale 在进入 attention/FFN 前使用近零恒等的平滑绝对界 `4.0`；raw/contracted modulation、QK、
FFN input 及 self/FFN/summary 的 raw/proposed/bounded/written/compression 均已透出。合成大幅调制测试
证明 forward 与普通 autograd 有限。尚缺 fresh 长跑跨过原 `batch 1170` 窗口，因此本条仍保留。

## O-02：全局裁剪在 NaN 前已让其余主图近乎停止学习

**级别：P0。类型：O-01 的全图训练后果，不是独立数值噪声。**

P1 不属于 `bottom.decoder`，因此不经过 V120 decoder-local clip。`batch 1160` 的 main global
clip 被 `5.95e6` 的 P1/canvas 梯度支配，post-global 后：

```text
P1             8.68e-1
canvas seed    4.95e-1
Observation    3.65e-2
G              7.92e-4
S              3.23e-4
W              7.39e-9
P2             3.46e-9
P3             2.43e-9
bottom decoder 1.31e-7
```

这些 finite batch 仍会执行 optimizer/scheduler/step。因此正式 NaN 之前，训练已经连续产生“几乎只
更新 P1/canvas、其余主图被冻结”的错误优化步骤；non-finite sentinel 只能终止最后一批，不能防止
这种有限但灾难性的 owner 独占。

**关闭边界：**O-01 的根因修复后，爆炸前窗口的 raw/postlocal/postglobal owner 比例必须恢复到
普通 batch 的量级，且 G/S/W/P2/P3/bottom 不再被压到数个数量级以下。不能把 per-owner clip 当作
长期梯度配额，也不能用拆 optimizer 掩盖错误 Jacobian。

## O-03：dynamic P1 的新 query-only 语义尚待真实运行验证

**级别：P1。类型：确定的所有权/适配不一致；净动作伤害尚未单独量化。**

Schema35 将 cached static P1 定义为 factual base，却让 dynamic P1 同时控制 P2 query 和 P3
precision，造成第二个动作写出口。Schema36 已把接口重命名为 `policy_query_residual`，实际消费者为：

- P2 query 读取 `action + static + dynamic`；
- P3 precision 只读取 static P1，并由当前 action query 调制；
- protected consequence、controlled transition、两层 V120 layer contract 和 bottom protected base
  只读取 `static + W effect`；
- event contract 也不直接读取 dynamic P1。

Schema35 的 dynamic/static 中位比约 `40-54x`、p90 超过 `114x`，但 Schema36 后两者分别是
query refinement 与 factual value，不能再用同一 RMS 直接宣称谁覆盖谁。真正需要验证的是：P1
query residual 是否仍长期贴住写入上限、是否使 P2 posterior 失去 typed/W 敏感性，以及移除 P3
直路后 precision/action 是否保持有效。

**关闭边界：**fresh run 中 query residual、P2 score/posterior、P3 static precision、transition、bottom
和 action 必须分别可定位；query residual 不得重新获得另一条 value 写出口，也不得长期使全局裁剪
由 P1 独占。不能仅凭 dynamic/static RMS 比关闭本条。

## O-06：W 有真实状态和监督，但仍明显低拟合 Teacher；实际 bottom 采用程度未知

**级别：P1。类型：可识别性问题与诊断缺口，当前不能称为 W 断线。**

W1/W2 的 near/far 因果边界、typed owner、camera axis 和 future loss 均真实存在。W working state
及 base interaction 也不为零，因此“W 没接上”不符合源码证据。但在普通 batch：

```text
predicted common/residual       约 0.133 / 0.028
Teacher common/residual         约 0.181 / 0.061
predicted/Teacher interval var  约 0.021 / 0.048
W adjacent cosine               约 0.97
```

Schema36 已修复 P2 的 selector/value owner 不匹配，但不能在 fresh 日志前宣称它解释了全部 Teacher gap。
同时，active V120 bottom 的源码证明 P3 lanes 和 protected consequence 可达；Schema35 compact log
却没有输出 `policy_delta_attnres`、protected-basis read 和各 lane 到 action 的采用指标，只有可达性，
没有实际采用证据。

**关闭边界：**先关闭 O-01～O-03，再用同 checkpoint 的 W common/residual、semantic/geometry
matched interventions，沿 P2 -> consequence -> bottom reader -> action 报告。若边界正确而 action
仍无净收益，归类为数据/Teacher 可识别性，不再继续改接线或增加 W gain。Status 只验证中性 loss
的恢复力，不再作为 action intervention。

## O-07：G3、learned flow 和 Teacher association 的独立贡献仍未识别

**级别：P2。类型：可识别性风险，不是当前确定性断路。**

- G1/G2/G3、G2 的 N=49 重新物化、G3 bounded conditional-K residual 均真实执行；G3 parent L1
  约 `4e-4`、assignment change 约 `1-2%`，但小幅不等于无用。
- flow confidence 已升到约 `0.26`，但历史 action intervention 很弱；当前首个 NaN 参数不能反推
  flow 是爆炸根因。
- Teacher dustbin 约 `0.46-0.49`、reliability 约 `0.24-0.25`、effective supports 约 `38`，说明
  association 分散；这可能来自真实小运动/弱可识别性，也可能仍限制 W target。
- object camera evidence 可变得非常集中，P2 camera effective count 约 `1.15`；尚不能区分正确视角
  选择和过早单视角化。

**关闭边界：**分别做 G3 owner/fact/action、flow zero/spatial-shuffle、DINO-key shuffle 和 camera
permutation/zero。必须先改变相应边界再讨论 action；不得通过放大 G3/flow、非零配额或额外 identity
loss 伪造使用。

## O-08：两个较低优先级的条件压缩/常量证据风险仍存在

**级别：P2。类型：多任务容量与 inherited null 风险。**

1. S 内部保留完整 Goal/History/G，但给 static P1 的 factual dock 仍把四个 Goal query 求 mean、
   history 只取最后 token。单任务可能被 interval context 覆盖，多任务/长历史时可能成为瓶颈。
2. bottom generic trajectory 输入为精确零，但 V120 `LayerNorm + Linear(bias=True)` 会把它变成
   trainable constant evidence。它可能是合法 null，也可能吸收 selector 质量。

**关闭边界：**先用冻结边界确认是否存在独立信息/动作采用。若 trajectory 只是 null，其 value 必须
精确为零、identity 放在 value 外；若 S 摘要丢失信息，只能用 query-based compact read，不能重开
视觉或恢复条件汤。

## O-09：当前测试与日志不能完整覆盖 active graph

**级别：P1（审查可靠性）。类型：确定的验证盲区。**

- active policy 使用 `RestoredV120ObservationCompiler` 和 `RestoredV120EvidenceBottom`；但部分
  structural tests 仍直接实例化 inactive `model/observation.py::CurrentObservationCompiler`。这些
  测试通过不能证明生产 Observation 路径正确。
- compact log 未记录 active bottom 对 protected consequence 与四个 P3 lanes 的实际读取/JVP。
- `best.pt` 按 normalized RMSE 选择，而当前主要行为门使用 physical RMSE；两者可能选择不同 epoch。

Schema36 已补齐 dynamic P1 内部诊断及生产 policy 的幅度/梯度测试；其余关闭边界不变：测试必须
实例化生产 policy 并从 serialized run context 锁定 active implementation；诊断只增加现有边界的
观测，不得为了测试再建平行实现。checkpoint 选择口径需显式统一或同时记录。

## O-10：Schema35 长跑被 P0 故障中止

**级别：P0 放行门。类型：实验无效。**

Schema35 epoch 1/2 physical RMSE 为 `0.0990/0.0931`；第二轮同 epoch 优于 V120 的 `0.0957`，
说明崩溃前行为并非整体失效。资源也正常：约 `11.75 GiB`、中位约 `1.92 s/batch`。但 epoch 3
`batch 1170` 已 non-finite，不能继续外推后期性能，也不能用前两轮好转掩盖优化路径已被 O-01/O-02
破坏。

**关闭边界：**修复 P0 后必须 fresh run；先复现跨过原崩溃窗口且 owner 梯度正常，再完成八轮，
比较 physical/normalized、first/tail、horizon、arm/gripper/event/motion、W/P2/P3 和后期反弹。
