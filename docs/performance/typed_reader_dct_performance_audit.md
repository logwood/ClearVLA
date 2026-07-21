# Typed Reader / DCT Performance Audit

更新时间：2026-07-16

## 范围

本记录针对当前 `policy/decoder.py`、`policy/controller.py` 的 typed reader + DCT action chart，
以及两份新训练日志。目标是区分真正有语义收益的计算和可以移出训练热路径的附加计算。

审计对象：

- `C:/Users/ASUS/.codex/attachments/19c7554a-0bc2-40d9-a9dd-d6008852e21b/pasted-text.txt`
- `C:/Users/ASUS/.codex/attachments/73f38a6c-5cb1-4f96-af7f-660b8518de0b/pasted-text.txt`

## 日志结论

两份日志分别表现为 6-step 和 3-step 路径。关键点如下：

| 路径 | batch 20 spb | batch 100 spb | batch 200 spb | batch 400 spb |
| --- | ---: | ---: | ---: | ---: |
| 6-step | 4.080 | 3.859 | 3.583 | 3.541 |
| 3-step | 3.187 | 3.306 | 2.958 | 2.997 |

1. 3-step 相比 6-step 明显更快，refine step 数仍然是首要的线性成本来源。
2. 两条路径在 batch 200 之后没有出现明显的 QR/contract 速度阶跃。当前 QR 是潜在后续风险，
   但不是这两份日志里最主要的瓶颈。
3. `hmread` 从约 `0.1` 增长到 `0.25--0.45`，`hmrdiv` 也从约 `0.02` 增长到 `0.06--0.16`；
   reader 的工作量和读取差异都在增加，不能把 reader 判定为无效旁路。
4. `hmswarp` 在 batch 200 后明显打开，说明频率坐标的可学习展开已经进入主路径。这个计算虽然有成本，
   但属于 DCT 表征所需的语义计算，不应通过删除 frequency reader 来换速度。
5. `hmcomp` 大约维持在 `7.7--8.0`，`hmwic` 大约维持在 `7.3--8.0`；当前没有证据表明 8 个控制槽
   已经完全退化成单一槽。`hmrdiv` 不是很大，但在增长，应该继续观察而不是强行加入竞争损失。
6. `hmraw/hmkeep` 在前 200 batch 基本保持满值，之后才出现轻微收缩。自适应深度尚未在早期承担明显计算削减，
   这解释了前期速度不会因为 controller 存在而自动变快。
7. `pflow/pfn` 在 200 batch 后开始出现可见差异，属于 contraction/aperture 开始介入后的正常观测点；
   不能仅凭某一个 batch 的 pflow 波动判定 reader 或 DCT 失败。

## 全量指标复核

这里不是只看 batch 20 和最后一个 batch，而是扫描了两份日志的全部训练行：第一份 21 行（到 batch 420），
第二份 22 行（到 batch 440）。

### 主动作、rollout 和事件

- 第一份：`pflow` 从 `1.945` 降到 `1.532`，全段均值约 `1.749`；第二份从 `1.900` 降到 `1.283`，
  全段均值约 `1.742`。中间存在 batch 100 附近的随机上冲，但整体不是平台。
- `pfn` 与 `pflow` 几乎同步，末端差异只有约 `0.007` 和 `0.004`，没有出现两条主流明显分叉。
- `afmd` 最终分别约 `1.53` 和 `1.26`，说明主动作/流形误差确实在下降。
- `rollout` 全段均值约 `0.316`，长期仍在 `0.24--0.42`；`rvar` 从 `3.4--3.6` 快速降到约 `0.02--0.04`，
  `rnorm` 也从 `7.6--8.8` 降到约 `0.05--0.08`。这表示 rollout 分支的内部状态先完成稳定化，但 rollout 任务本身仍在
  `0.3` 附近，不能把 `rvar/rnorm` 的下降误认为 rollout 监督已经解决。
- `first8` 和 `tail` 都在下降，没有出现 tail 单独爆炸；早期 tail 通常比 first8 高，后期差距缩小。
- `event` 从约 `0.74--0.79` 下降到 `0.30--0.38`，中间最低到约 `0.11--0.22`，波动明显。夹爪事件仍是当前
  最不稳定的任务项之一，需要单独看验证集和事件覆盖，不能用 arm 的 pflow 代替判断。
- `stdr` 从约 `0.15` 增长到 `0.83--0.87`，`dnratio` 从约 `0.10` 增长到 `0.43--0.50`；这说明预测/去噪残差在变得更活跃，
  目前没有和 global grad 一起爆炸，但应继续观察它与 event/tail 的关系。

### 坐标、DCT 和频率展开

- `anull/gnull` 全程为零，`hmsgeo` 约 `1e-5`，`hmtan` 约 `1e-6`；当前没有看到 DCT/Parseval 坐标错位或明显 null 泄漏。
- `hmchart/hmspec` 全程保持合法状态，说明 chart 选择和 spectral state 没有丢失。
- batch 200 以前 `hmswarp` 接近零；之后上升到约 `1.6--2.3`，说明频率坐标变形确实被 controller 使用。
  这不是可以直接删除的“无用计算”，但要检查它是否长期贴近 shift limit，以及 `hmkerr` 是否持续扩大。
- `hmkerr` 在后段约 `1e-4`，`hmgerr` 约 `1e-5`，目前是小量但不再是严格零值；应作为频率坐标约束的健康指标保留。

### controller 和 reader

- `hmdu/hmur` 整体上升，说明 recurrent controller 状态变化没有冻结。
- `hmread` 从约 `0.1` 上升到 `0.25--0.45`，`hmrdiv` 从约 `0.02` 上升到 `0.06--0.16`；typed reader 的读取内容和
  读取差异都在形成。
- `hmcomp` 仍约 `7.7--8.0`，`hmwic` 约 `7.3--8.0`。这只能说明读取负载和有效 slot 数没有明显坍缩，不能单独证明
  8 个 slot 已经拥有 8 种独立语义；需要结合 `hmrdiv` 和 frequency ownership 一起看。
- `hmcos` 从约 `0.996` 降到 `0.976--0.988`，`hmbdot` 转为约 `0.075--0.11`，`hmcan` 从约 `0.49` 降到 `0.43--0.45`。
  分支之间仍有较强同向性，且后段更明显；这属于潜在横向冗余，但不是当前速度主因。
- `hmnf` 下降、`hmsf` 小幅上升、`hmlf` 基本稳定，stage 相关份额仍偏弱，尚不能说 stage memory 已承担主要职责。
- `hmselgrad` 全程为 `0`，`hmexitgrad` 也全程为 `0`。如果当前实验本来就是固定 schedule/关闭 learned exit，
  这是配置结果；如果预期 learned stage selector 或 learned dwell 正在训练，则这是尚未接入主反传的明确问题。
- `hmctrlgrad` 前期约 `0.13--0.56`，batch 100 后多数降到 `0.01--0.03`；controller 前向仍在工作，
  但其可学习控制量的梯度正在变弱。不能只看 `hmcomp≈8` 就认为 controller 学得充分。
- `hmfunc` 在 batch 200 后可升到 `6--9`，说明 function reader 不是空转；同时要监控它是否开始成为过强的控制旁路。

### workspace、梯度和稳定性

- workspace 的 `olupd` 从约 `0.5--1.1` 增长到 `3--5.7`，`ohsupd` 从约 `3.1--3.6` 增长到 `3.2--3.9`，
  说明 workspace 更新幅度在扩大。这是有效学习信号，但也意味着后段 workspace 可能重新成为幅度主导者。
- `ohsret` 基本稳定在 `0.716--0.719`，promotion 没有失控；`ohprom` 反而从约 `0.10` 降到 `0.05--0.08`，
  需要结合实际 role coverage 判断是否过度保守。
- `hmdgrad/hmvgrad` 没有持续爆炸；global `grad` 从约 `2.8--3.1` 下降到约 `1.0`，训练数值稳定。
- `hmbgrad` 和 `hmbasegrad` 在后段比早期更高，说明 MMDiT 主块并非没有收到梯度。
- `hmcopgrad/hmcgrad` 在 contraction 打开前为零，打开后只有约 `1e-3`；这是符合“先 identity、后轻微收缩”的状态，
  但也说明 contraction 目前还没有显著改变主优化方向。
- `d_shuffle` 近零、`contrast` 约 `0.03--0.05`，当前不能从这两个指标证明状态反事实监督很强或很弱。

## 已确认的热路径开销

### P0：operation candidate probe 重复执行 MMDiT

`scripts/current_v85_unified_controller.sh` 默认打开：

```bash
HIERARCHICAL_MMDIT_OPERATION_CANDIDATE_PROBES=1
HIERARCHICAL_MMDIT_OPERATION_ROUTE_LOSS_WEIGHT=0.05
```

`decoder.py::_probe_operation_candidates` 在真实 block 更新之后，又对当前 block 和下一 block 的合法
stage 做无梯度 block forward。无梯度只消除反传和大部分激活保存，不消除矩阵乘法。

在 `3 block x 2 stage` 的当前配置中，固定 3-step 路径每个样本大约额外产生 `4 + 4 + 2 = 10` 个候选
block 行；6-step 路径约额外产生 `20` 个候选 block 行，而真实更新只有 3 或 6 行。

此外，候选结果回填使用逐元素 `.item()` 的 Python 循环，位置在 `decoder.py::_probe_operation_candidates` 末尾。
这会制造多次 GPU/CPU 同步和小 kernel launch，是比单纯无梯度 forward 更严重的实现开销。

处理顺序：

1. 先用 `candidate_probes=0`、`operation_route_loss_weight=0` 做速度 A/B。
2. 如果保留 route 监督，改成张量化回填，并对候选做采样或只评估一个合法替代项，不枚举全部候选。
3. 不要为了保留一个很轻的 route loss，长期支付整条候选 MMDiT 路径的成本。

### P1：详细 controller 仪表进入每个 forward

`controller.py` 当前每个 refine step 都会执行：

- FP32 Gram 矩阵和 `torch.linalg.eigvalsh`；
- 完整 output attention 权重、entropy、slot load 和 diversity；
- recurrent attention/ownership 的 detach、stack 和聚合；
- spectral ownership 的竞争统计。

这些数值主要用于日志和审计，并非主训练目标。尤其 `output_attn(..., need_weights=True)` 会物化完整
attention 权重，也不利于使用更高效的 attention kernel。

建议将仪表分成两层：

- 每个 batch：只保留有限的标量安全检查和真正参与 loss 的值；
- `log_every` batch 或单独 probe batch：计算 eigvalsh、attention map、quantile、ownership 全套诊断。

### P1：反传后的梯度诊断重复遍历参数

`policy_runtime_v39.py::_attach_grad_diagnostics` 在每个 batch 反传后，重复扫描 decoder、blocks、workspace、
contraction、controller 及多个重叠子组的梯度。当前日志需要这些信息，但不需要每个训练 batch 都完整计算。

优先做一次参数分组聚合，或者只在日志 batch 计算详细分组梯度；不要改变实际 gradient clipping 和 optimizer 行为。

### P2：detached velocity prediction 被重复调用

`decoder.py::_detached_velocity_prediction` 在 refinement 初始、每一步之后以及 candidate probe 中重复执行 velocity head，
并在 spectral path 中执行 DCT decode。24x24 的 DCT 单次很小，但被重复调用后会积累 kernel launch 和 FP32 einsum 开销。

其中用于 controller feedback/exhaustion 的 prediction 不能直接删除；纯诊断 prediction 可以降频。candidate probe
关闭后，这部分会自然减少一大截。

### P2：contraction 启动后的 QR

`decoder.py::prepare_contraction_factors` 会为每个 block/branch 的 contraction bank 重新构造正交 basis。当前配置是
3 个 block、5 个 branch，每个 bank 还有多个 stage。两份日志没有显示 batch 200 后出现明显速度阶跃，但应在后续长跑中
单独监测 batch 200 前后的 spb。

如果未来确认 QR 成为瓶颈，再考虑正交参数化或可复用的 retraction；不要现在为了猜测而降低 rank 或改变 contraction 几何。

## 不应作为“性能优化”删除的部分

1. 24 个 frequency query。DCT 后 token 是频率坐标，不再是一 token 对应一个时间点；频率 reader 是恢复局部频率语义的核心。
2. frequency local coupling 和 aperture/warp。它们负责把粗到细的频率展开变成可学习的连续路径。
3. typed source / operator / exit / spectral reader 的职责分离。当前 `hmread`、`hmrdiv`、`hmswarp` 的演化说明它们正在工作。
4. DCT 的正交 chart 和 arm/gripper 分治。真正的问题是重复调用和外围诊断，不是 DCT 表示本身。

## 推荐执行顺序

1. 先做 3-step、同一 GPU、同一 batch 配置的 `candidate_probes=0` A/B，记录 batch 20--100 的平均 spb。
2. 将候选回填向量化；若 route 监督没有明确收益，保留默认关闭。
3. 将详细诊断改为按 `log_every` 采样，不改变 loss、reader、DCT 或 controller 前向语义。
4. 再优化 DCT reader 的静态 source K/V 缓存和 attention 权重物化。
5. batch 200 以后再判断 contraction QR 是否值得重构。

## 目录整理边界

- `scripts/current_v48_...` 到 `current_v87_...` 是实验谱系，不做删除；它们可用于回放和归因。
- 当前主线代码和新 `current_v87_spectral_controller.sh` 不移动。
- 仅将已经完成阶段的顶层设计文档归档到 `history_design/archive/`，避免根目录继续堆积历史计划。
- `.worktrees`、`legacy`、`history_log`、`policy_semantic_map` 保留，它们分别承担重构、历史代码、日志和语义地图职责。
