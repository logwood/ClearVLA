# ClearVLA 当前主线纯问题账本

更新：2026-08-21

当前源码身份：Schema27 `object_intent_dynamics_323`。行为比较锚点仍是
V120 `long`、提交 `0b92d359a2889a0a1b1eba256007c00ccbc54f3c` 与本地
快照 `.audit/v120_exact_source_0b92d359/`。V120 是行为锚点，不是正确性公理。

本文件只保留尚未被证据关闭的问题。已完成的实现决策与不变量只写在
`00_CURRENT_ARCHITECTURE_CONTRACT.md`；问题一旦被源码和实验共同关闭，就从
本文件删除，不保留版本堆叠史。

## 记账规则

- 源码能证明的旁路、轴丢失、非零默认值或错误梯度生命周期，才记为确定性故障。
- 曲线相关性不能单独证明因果；没有冻结干预时，明确写“动作影响未知”。
- 张量存在、梯度非零、loss 下降都不等于该边界被策略使用。
- 不用新 block、额外外部 loss、gain、quota、hard gate、熵/多样性目标或人工梯度掩盖问题。
- Schema27 正式比较必须 fresh run，并与 V120 对齐数据、seed、batch、动作归一化和验证口径。

## U-01：global-K 是否形成可复用对象身份仍需实验确认

**类型：行为风险。置信度：高。对动作差距的独占因果：未知。**

Schema27 已静态关闭此前能确定的三个缺口：grounder 使用独立的当前 DINO
监督、semantic/appearance/geometry 在 K 绑定前共同参与一个物理 K+null
竞争、detached existence 进入在线可选性但不进入 loss mask。仍不能由源码
保证 K 会在当前单任务数据上形成稳定、动作相关的对象身份；Teacher 仍由同一
套当前 G 槽自举，也不能提供外部 object ID。

首个验证边界：

- `object_grounding_object_content_pair_cosine`
- `object_grounding_object_chart_pair_overlap`
- `object_grounding_prebind_*_l1`
- `object_grounding_existence_mean / null_mass / mass_conservation_error`
- global-K permutation equivariance 与冻结 K-mean/K-permute 动作干预

只有对象区分在训练与验证中都改善，并且对应干预先改变 G/S/W 边界、再改变
动作，才能关闭本项。单独 reconstruction 下降不能关闭。

## U-02：S 的四区间与 typed future 现在可识别，但真实条件使用尚未证明

**类型：行为风险。置信度：高。**

Schema27 已删除自由坐标的 future recognizer hidden；public carrier 直接预测
四区间 future state，原样进入 W 的 semantic/appearance/geometry typed value
分别预测 Teacher 的 semantic/status/transport 字段。源码已经保证 target 口径、
K/type 轴和消费者边界一致，但仍可能只学习数据集平均未来或固定区间模板。

必须联合观察：

- public 与三类 typed prediction/target RMS 及各自 loss；
- raw interval variation 与 `condition_centered_interval_variation`；
- goal/history/object innovation；
- goal/history/G typed zero/shuffle 对 S 边界、W 字段和 action 的分层影响。

固定 interval identity 产生的变化不计作条件感知。若直接 target loss 下降而
condition-centered variation 和干预仍近零，本项继续保留，不能靠强迫多样性关闭。
该 condition-centered 指标会减去每个 interval 的 batch 均值，所以 batch=1
smoke 中按定义恒为零；只能用正式 batch=8 或显式条件干预解释，不能把 smoke 的
零值当作 S 坍缩证据。

## U-03：W 的字段所有权已静态成立，但未来区分和动作效益仍未证明

**类型：行为风险。置信度：高。**

W1/W2 的 public working state 现在只能乘性调制对应 typed sidecar，不能凭空
生成 semantic/transport/status value；三类 sidecar 直到匹配输出头前均不相加，
visibility/persistence 也没有自由 bias。进入 P2 的仍只有直接监督的
`FutureObjectDynamics`。这些事实关闭了“W 内部重新公共化”的确定性接线故障，
但不能证明 W 已学到比平均未来更细的四区间后果。

必须检查：

- prediction 与 Teacher 的 interval/object cosine、condition-centered variation；
- 四区间 semantic/transport/status 的 target-normalized error；
- typed sidecar 分项 RMS 与梯度；
- effect zero/shuffle 是否改变 P2、consequence，最终 action 的置信区间是否离开零；
- zero W effect 是否仍改善动作误差。

Teacher 由 G 自举（U-01）会限制 W 的上限，但在没有干预证据前，不把两者写成
同一个因果问题。
W 的 condition-centered variation 同样在 batch=1 时按定义恒为零；该口径只在
batch>1 或显式干预下用于判断未来是否随条件变化。

## U-04：P1 动态 action self-write 仍可能压过静态精细事实

**类型：V120 祖传结构风险。置信度：中高。**

Schema27 保留了 V120 的 24 query、N=49、四 glimpse 和 3×3 microgrid，也没有
改动 bottom。P3 已删除 protected fact 的重复 optional 编码，P2 null 候选数先验
也已校正；但 P1 动态路径本身仍可由 action query 写入较强的 policy/FFN delta。

需要把以下量与 V120 同年龄比较：

- `p1_protected_detail_rms`
- `p1_dynamic_delta_rms`
- `p1_policy_self_written_rms`
- `p1_policy_ffn_written_rms`
- `p1_spatial_variation`、microgrid value RMS
- raw/detail/address 干预到 completed P1 与 action 的链式效应

只有当静态视觉干预在 completed P1 前已经消失，或动态自写显著超过 V120 并伴随
动作退化，才升级为下一轮结构修复。本轮不预先拆 P1。

## U-05：后期 gripper 与中远程回弹仍是未归因的泛化问题

**类型：完整长跑问题。置信度：高；结构归因未知。**

V120 及后续若干 schema 都出现过 epoch 7/8 的 gripper、tail 或中远程反弹。
当前源码没有按 epoch 启动的新结构分支，因此不能把它解释为“后期状态切换”。
候选原因包括任务事件稀疏、动作场过拟合、S/W 平均未来和 bottom 补偿，但现有
证据不足以排序。

关闭条件是完成八个 epoch，并同时比较：

- train action/native/arm/gripper 与全部 validation RMSE；
- first/first8/tail、1-4/5-12/13-24 bands；
- gripper event/hold、event/motion head；
- G/S/W/P 边界和 raw/postlocal/postglobal 梯度；
- 最佳点、最终点和八轮均值，而不是只取 epoch 1 或单一最低 RMSE。

若 Schema27 的结构边界与冻结干预都正常而该反弹仍复现，优先归类为数据/目标
可识别性与泛化问题，不继续改接线。

## MIP-01：MIP 与 B-spine 的性能研究尚未进入主线

**类型：前瞻研究。状态：只记账、未改模型源码、未分配 MIP 子 schema。实现父版本必须是放行后的 Schema27；论文直接证据只覆盖 MIP-2，K>2 是实验扩展。**

整个 ClearVLA 可视为层级、对象中心、隐状态驱动的 WAM。MIP 的合法所有权在
bottom 最终动作生成边界，但一次 stage 的合法执行单元不是最后一个 DiT block，
而是完整 dynamic B-spine：

```text
static WAM stem（每 observation 一次）
  policy.encode_online
    -> Observation/G/S/W
    -> cached P1 fact / G3 transition source / role table

dynamic B-spine（每 action stage 一次）
  policy.velocity(noisy_physical, time, cache)
    -> action query（不直接消费 time）
    -> dynamic P1（直接消费 time）
    -> P2/P3 -> controlled transition -> restored_bottom
    -> time_domain_mmdit.EvidenceLatentMMDiTActionDecoder
    -> 3 × TimeDomainMMDiTBlock -> action_norm
    -> velocity / event / motion heads
```

active readout 位于 `v120_core/time_domain_mmdit.py:3090-3092`；
`v120_core/decoder.py::_run_refinement/_velocity_prediction` 不是当前主路。MIP stage
loop 必须留在 decoder 外层，复用完整 `policy.velocity`，只把同一 18-D physical
field 带到下一 stage；不得重建 G/S/W，也不得在 stage 间 decode/re-encode。

### 首版训练适配

首版保留 physical velocity head，不改 direct-action regression，不增加 stage
embedding。`ActionQueryEncoder` 校验后会 `del time`；time 直接进入 dynamic P1
和 active Evidence decoder，并通过 P1 间接影响 P2/P3/transition。测试必须分别
确认 query 对 time 不变、dynamic P1 与最终 velocity 对 time 可变。

训练节点采用解耦 teacher interpolant。给定 target physical action `a`：

```text
z ~ N(0,I)                    # 完整 batch，owned train_flow_generator
i ~ Uniform({0,...,K-1})      # 整个 batch 共用一个 scalar stage
t = stage_times[i]
source = 0 if i==0 else z
x_t = (1-t)·source + t·a
u_t = a-source
prediction = policy.velocity(cache, x_t, t)
clean_estimate = x_t + (1-t)·prediction
```

首版选择“每 batch 一个 stage”，不是每样本 mixed-stage。当前
`execution_value_terms` 用全 batch 的 `target_spread`、`reliability_scale`、
`normalization_scale` 做非线性归一化；同 batch 混合 stage0 的 `a` 与后续
`a-z` 会互相改变 controller loss。每样本 categorical 只有在 execution 按
stage 分组重构后才可作为 fidelity 选项，而 batch 8、K=4 时每组约两样本会很噪。

设 stage loss 为 `L_i`、目标权重为 `w_i`、抽样概率为 `p_i`，规范目标是
`Σw_iL_i/Σw_i`，单次无偏估计为 `w_i/(p_iΣw_j)·L_i`。均匀抽样、全权重为 1
时 sample weight 就是 1；K-stage 全展开必须除以 K。首版不提供
`action_stage_weights`，防止把含静态 `history_proposal_loss` 的整个 action group
一起重权。Observation/G/S/W/Teacher/JEPA/proposal 仍每 batch 一次，dynamic
action/execution 只贡献一次。

每 stage 指标按 `numerator + sample_count/support` 聚合；未抽到该 stage 不能填零。
stage 间 gradient cosine 只在低频固定诊断 batch 上用 K 次 dynamic forward 和
`autograd.grad` 测量，诊断 generator 不得扰动正式 validation RNG。

### 递推、endpoint 与初始化

对 `v_i` 定义 `a_hat_i=x_i+(1-t_i)v_i`。部署必须显式区分：

```text
euler:         x_next = x_i + (t_next-t_i)·v_i
clean_rescale: x_next = t_next·a_hat_i
```

二者在 MIP-2 的 `[0,.9]` 上等价，在 K>2 且模型有误差时不等价。首轮矩阵为：

| 身份 | `t_call` | transition | 证据 |
|---|---|---|---|
| Schema27 五步对照 | `[0,.2,.4,.6,.8]` | Euler | 当前 V120-compatible runtime |
| `mip2_velocity_reference` | `[0,.9]` | 二者等价 | 论文/官方端口直接证据 |
| `mipk_clean_rescale_experimental` | `[0,.55,.75,.9]` | clean-rescale | 主 K>2 候选，未验证 |
| `mipk_euler_experimental` | `[0,.55,.75,.9]` | Euler | 同节点求解器控制，未验证 |

K=4 的 `.55` 节点仍含 `45%` Gaussian source；它既改变部署步长，也改变训练
腐化强度。暂不把最后 action call 推到 `.95`，也不把“后期步长越小越好”当先验。

离散 action nodes 最高只到 `.9`，而当前部署在 `t=1` 读取 event/motion heads。
若无匹配训练，这个 endpoint 是 OOD。首版建议 `t1_explicit_supervision`：用 clean
target field 在 `t=1` 再做一次完整 dynamic forward，但只训练 event/motion heads，
不重复 velocity/action/execution 或静态损失，并保持原有一次 head 预算。
`last_action_stage` 是独立 runtime ABI/ablation，训练和部署必须同时读取末 stage
heads。NFE 分别报告 action-update=`K`、endpoint-head=`1`、total dynamic
B-spine=`K+1`；论文 MIP-2 NFE=2 不含 ClearVLA 的额外 endpoint。

初始化至少区分 `physical_origin_zero`、`encoded_native_zero`、`encoded_hold`。
18-D physical 全零 decode 后含约 `0.25×action_state` 的 delta 路历史项，因此
“tensor 为零”不等于 native 零动作。严格 MIP-2 reference 按官方端口消耗一次
noise draw 后覆写为零；`no_draw_zero` 是另一 runtime RNG ablation。

### 身份、修改面与放行

typed config 至少拥有：

```text
action_policy_mode
action_training_distribution     # V120 mirrored Beta | MIP discrete stage，互斥
action_stage_times
action_stage_sampling            # continuous | batch_uniform_categorical
action_stage_transition          # euler | clean_rescale
action_initialization
action_endpoint_head_source
action_runtime_rng
```

`inference_steps` 只由 canonical schedule 推导或交叉验证；tuple/list round-trip、
config digest 和非法组合必须 fail-closed。任何 MIP mode 可执行的同一提交都必须先
分配子 schema、training/runtime ABI、正式 config、run-context 字段与 exact-resume
拒绝；不能先运行、后补身份。Schema27 checkpoint 不得 exact-resume 为 MIP。

预计修改面为 `config.py`、`training/losses.py`、`training/engine.py`、`train.py`、
`runtime/sampling.py`、`runtime/evaluation.py`、`runtime/logging.py`、
`audit_policy_logs.py`、`manifest.py` 与对应测试。默认不改 G/S/W、P1/P2/P3、
transition、restored bottom、active Evidence MMDiT、velocity head、CVAE/workspace
或 execution controller 的 forward 语义。bottom-only 迁移若作为消融，run context
必须记录初始化模式、来源 checkpoint 身份/hash 与实际 loaded prefixes；主结果仍
采用 fresh end-to-end training。

测试必须覆盖 Schema27五步 RNG不变、schedule tuple/list round-trip、K=2两种
transition数值等价、K>2身份可区分、static stem一次/dynamic K次、endpoint训练与
部署source一致、静态损失不随K放大、stage/noise exact restore、execution ablation
复用同一显式 initial physical field，以及诊断不改变正式validation结果。

放行顺序固定为：先完成 Schema27 当前实现、fresh smoke 与 early/late gate；再
冻结 MIP-2/K>2 公式和测试向量；随后在一个原子新身份中实现 batch-stage training、
matched endpoint 与 deployment transition；最后以同数据、seed、batch、optimizer、
预算对比 Schema27 五步、MIP-2、K>2 clean-rescale 与必要 Euler control。性能优先
于步数，只有逐 stage gain、rollout gap、endpoint、gripper rebound 和梯度冲突账
共同支持时才调整 K/节点。

## 当前放行顺序

1. 本地完整回归、参数/optimizer ownership、BF16/FP32 与生命周期静态审查。
2. fresh batch-8 smoke；确认参数清单、Teacher 次数、五步+endpoint 调用、显存和有限梯度。
3. 对齐 V120 的 batch 约 2200 早期审计，重点核对 U-01 至 U-04 边界。
4. 早期没有结构退化后才跑八个 epoch，用 U-05 的完整口径判断。
5. 若结构边界正确但动作无增益，停止继续堆接线，将问题转为数据/监督可识别性研究。
6. Schema27 全部 gate 关闭后，MIP-01 才进入独立子 schema；不得借 MIP 研究绕过当前 scope lock。
