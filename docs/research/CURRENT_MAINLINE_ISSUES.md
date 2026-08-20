# ClearVLA 当前主线纯问题账本

更新：2026-08-20

本文件只记录仍未被证据关闭的问题，不记录已完成修复的历史。当前源码身份为 Schema26 `object_intent_dynamics_323`；行为基准固定为 V120 `long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 和本地完整快照 `.audit/v120_exact_source_0b92d359/`。目前可比较的新主线日志仍来自其 Schema25 父版本、截至 epoch 1 batch 2180；因此下面所有 Schema26 项目都是放行条件，不是已经宣称恢复的结果。

## 记账规则

- 源码数据流可直接证明的问题记为“确定性故障”；训练曲线只能证明异常而不能唯一定位时记为“行为问题”。
- shape 正确、张量非空、梯度非零和模块有名字，都不等于语义接通。
- 不把相关性写成因果。没有冻结干预或等价边界证据时，明确保留“动作影响未知”。
- 已经静态关闭的问题从本文件删除；对应架构决定只写入 `00_CURRENT_ARCHITECTURE_CONTRACT.md`。
- 不通过增加 block、loss、gain、quota、hard gate、entropy/diversity 约束或人工梯度来关闭本账本。

## 当前对齐基线

Schema25 与 V120 在 batch 2180 的同年龄比较为：

| 指标 | Schema25 | V120 | Schema25 相对落后 |
|---|---:|---:|---:|
| action flow | 0.080623 | 0.062641 | 约 28.7% |
| native flow | 0.109999 | 0.093019 | 约 18.3% |
| decoded action | 0.019221 | 0.017926 | 约 7.2% |

差距主要累积在中间动作场/组织链，而不是只出现在 decoded head。Schema26 已修正两个明确的数据流边界，但尚无新实验数据，不能据此宣称上述差距已经关闭。

## P1-01：Schema26 的早期动作恢复尚未得到行为验证

**类型：放行阻塞型行为问题。置信度：高。根因分摊：尚未完成。**

Schema26 源码已经让 P1 与 transition 共用同一个最终 G3 rollout，并移除了 transition 的 public-chart 复制与 interval identity 伪轴；同时关闭了 S typed evidence 经 CoarseAction 重复进入 W 的第二条路径。这两项修复具有确定的语义正确性，但动作收益仍必须由 fresh run 证明。

关闭条件：

- 在同 seed、数据、batch、normalizer 下，与 V120 对齐到 batch 2200，而不是与 Schema24/25 对齐；
- 比较 action flow、native flow、decoded action、arm、gripper，以及 G3 anchor variation、transition/source 梯度；
- 没有任一核心指标继续远离 V120，且总体关闭 Schema25→V120 差距至少 50%；
- 若精确 G3 source 已到达 transition 但动作仍无改善，应把剩余差距重新分配给 S/W 或优化可识别性，不能再次改写 bottom。

## P1-02：S 的四区间 typed 可识别性仍需新日志确认

**类型：父版本已证实的弱边界；Schema26 源码修复待行为验证。**

Schema25 在 batch 2180 的 typed object variation 尚可，但 interval variation 很弱：

- public interval variation：`0.1054`；
- semantic / appearance / geometry typed interval variation：`0.00287 / 0.00160 / 0.00151`；
- 对应 object variation：`0.223 / 0.115 / 0.071`。

这说明父版本更像“对象可区分、区间近公共”。Schema26 使用同一组 bias-free typed 投影分别读取公共分量与零均值区间差分，再做有界组合；完全相同的四区间不会被人工制造差异。该结构边界已经可执行，但不能把“新增了 differential metric”当作恢复证据。

关闭条件：

- 分别记录每种 type 的 `common_score_abs`、`differential_score_abs`、selector null probability、object variation 与 interval variation；共同的事实缺失只记录一次 `typed_fact_unsupported_fraction`，并确认 differential normalization denominator 不低于 0.25；
- 真实区间变化应先改变 S→W typed 边界，完全相同的 interval carrier 仍应产生精确零 differential；
- typed interval variation 的恢复不能只来自 learned interval identity，也不能以固定多样性目标强行抬高；
- 必须分开报告边界变化和最终 action 变化。

## P1-03：W 对四个 future interval 的解析度显著低于 teacher

**类型：已证实弱边界。置信度：高。根因分配：未完成。**

Schema25 batch 2180：

- W prediction interval variation：`0.03891`；
- teacher interval variation：`0.11363`；
- W 只保留约 34% 的 teacher 区间变化幅度；
- W2 adjacent interval cosine：`0.97419`；
- semantic / appearance / geometry 写入 W 后的 interval variation：`0.00145 / 0.00091 / 0.00093`。

W 具有四个 interval、K object、W1/W2 因果身份和直接 future loss，因此不能简化为“没有监督”或“没有接入”。Schema26 只修复了上游 typed 公共化的一处来源以及重复 ingress，没有重写 W；若新日志中 W 仍远弱于 teacher，这将成为独立的 W 可识别性/优化问题。

关闭条件：

- 同年龄比较四区间 prediction/target variation、adjacent cosine、逐区间 target-normalized error；
- 分别干预 S typed 边界与 W field，不能用一个合并 RMS 宣称二者同时恢复；
- W effect zero/shuffle 必须先改变 P2/consequence，最终 action 影响另行报告；
- reliability、validity 或 default geometry 的变化不得冒充 semantic successor/delta 学习。

## P1-04：epoch 7/8 的 gripper 与中远程反弹仍未归因

**类型：跨版本行为问题。Schema26 尚无对应后期数据。**

- Schema24 physical RMSE 在 epoch 6–8 为 `0.08008 / 0.08193 / 0.08218`，回退主要来自 gripper 与 5–24 步；arm 相对稳定。
- V120 自身也有较小的末期反弹（约 `0.0793 → 0.0814`），因此“存在反弹”不是新 S 独有故障。
- 现有相关性不能证明 P2 null、effect RMS、S typed 或 W 公共化中的任一项是唯一根因。

关闭条件：比较最佳验证点与最终点，并拆分 first/tail、1–4/5–12/13–24、arm/gripper、event/motion。若 Schema26 的结构边界健康而反弹仍存在，应转为泛化/数据可识别性问题，不能继续用接线补丁解释。

## P2-01：Schema26 的显存、吞吐和生命周期尚未实测

**类型：运行时放行条件，不是已知回归。**

删除未消费的 `future_address` 在线 grid-sampling 后，理论上 W 的静态计算与显存只会减少；移除 no-op proposal validation 也不应增加训练成本。但当前没有生产 GPU 数据，不能用静态推断替代测量。

关闭条件：

- 本地 BF16 微型 forward/backward 小于 8 GB；
- 生产 batch 8 总进程显存不超过 22 GB；
- Teacher 每训练 batch 一次、部署零次；Observation/G/S/W/静态 P1 每 observation 一次；
- 五步采样只重复动态 P1/P2/P3、transition 与 bottom，endpoint head forward 不更新 action；
- 与 V120 比较 seconds/batch、epoch wall time、allocated/reserved/process peak，不能只引用 preflight 峰值。

## P2-02：主线目录仍含未激活的重复实现

**类型：维护性问题；没有当前运行时影响证据。**

`clearvla/mainline/model/bottom.py`、`observation.py` 等文件仍存在与 active exported modules 并存的旧实现。当前 import closure 与实例化路径使用 restored modules，因此这不是本轮性能故障；但以后只按类名搜索容易把未激活代码误判为主路。

关闭条件：只有在 import closure、serialized manifest、checkpoint ABI 和完整回归测试均证明无消费者后才能清理。本轮不为追求目录整洁而删除它们。

## 当前放行顺序

1. 通过完整 CPU 回归、typed/anchor 等变与零值测试，以及 manifest/checkpoint/RNG 测试。
2. 运行 fresh batch-1 smoke，确认 dtype、finite gradient、五步部署与生命周期。
3. 运行 fresh batch-8 早期实验，对齐 batch 2200 与 V120；首先判断 P1-01/P1-02/P1-03。
4. 只有早期 gate 通过后才跑八个 epoch，并判断 P1-04。

在这些行为边界关闭前，不宣称 Schema26 已恢复或超过 V120；也不再扩散修改到 P1、P2、P3、transition core、CVAE、workspace、Evidence MMDiT、execution 或数据管线。
