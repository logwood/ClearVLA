# Gated-LoRA MMDiT 分块 Refine 实施计划

> 版本：v1.0  
> 日期：2026-07-13  
> 状态：设计与验证计划  
> 核心目标：让每一次 refine 对应一个真实、可解释、可独立注入条件的网络块，并用 gated-LoRA 限制每个块相对于共享 MMDiT 的参数修改子空间。

---

## 1. 背景与设计动机

现有多步 refine 如果反复调用同一个黑盒 refiner，会产生三个问题：

1. 不容易解释第 1、2、3 次 refine 分别做了什么；
2. 不容易为不同 refine step 定制条件入口；
3. 各步参数职责混合，消融时只能删除整个循环或修改统一模块。

本计划将每一次 refine 显式展开为独立网络块：

\[
A_0
\xrightarrow{R_1}
A_1
\xrightarrow{R_2}
A_2
\xrightarrow{R_3}
A_3
\]

其中，每个 \(R_k\) 都具有：

- 明确的条件读取权限；
- 独立的 gated-LoRA 参数；
- 独立的残差输出 \(\Delta A_k\)；
- 可记录的 gate、有效 rank 和修改幅度；
- 可单独删除、替换或改变条件接口的真实模块边界。

### 1.1 本计划的目标

- 通过低秩参数增量限制每个 refine block 的参数自由度；
- 通过显式 block 边界提高结构可解释性；
- 允许不同 refine block 注入不同条件；
- 在推理时保留动态 gated-LoRA，不将其静态合并进共享权重；
- 保持 Workspace、Low-level Evidence、Stage Memory 和 Action Update 的因果边界；
- 为逐块消融、条件归因和论文叙事提供直接接口。

### 1.2 明确不作为目标的事项

- 不以减少 FLOPs 或推理延迟作为主要目标；
- 不以 LoRA 作为模型压缩或部署加速手段；
- 不通过无结构权重稀疏制造难以解释的零值参数；
- 不允许 gate 或 LoRA 把 Action 信息写回 Workspace；
- 不使用通用 loader/factory 隐藏三个 refine block 的真实结构。

推理耗时可以记录，但不作为该方案是否成立的主要判据。

---

## 2. 核心设计结论

### 2.1 共享能力与分步差异分离

所有 refine block 共享完整的 MMDiT 基础能力 \(\theta_0\)，每一步仅通过自己的低秩参数增量形成差异：

\[
\theta_k = \theta_0 + \Delta\theta_k
\]

对某个线性映射 \(W\)，第 \(k\) 个 block 使用：

\[
W_k = W_0 + \Delta W_k
\]

\[
\Delta W_k = U_k\operatorname{diag}(g_k)V_k
\]

其中：

- \(W_0\)：共享的完整 MMDiT 权重；
- \(U_k,V_k\)：第 \(k\) 个 refine block 独立的 LoRA 基；
- \(g_k\)：由该 block 条件生成的 rank-direction gate；
- \(\|g_k\|_0\)：该样本在该 block 中使用的有效 rank 上界。

这意味着低秩限制作用于“第 \(k\) 步相对于共享能力能够改变什么”，而不是把整个 MMDiT 限制成低秩网络。

### 2.2 Rank 与修改强度必须分开控制

rank 只描述可以沿多少个独立方向修改参数，不代表修改一定微弱。一个 rank-1 更新仍可能具有很大的谱范数。

因此，每个 block 同时设置三类控制量：

| 控制量 | 含义 | 建议形式 |
| --- | --- | --- |
| Block gate \(s_k\) | 是否采用本次 refine | 标量，hard/soft gate |
| Rank gate \(g_k\) | 激活哪些低秩修改方向 | rank group 或逐方向 gate |
| Residual scale \(\gamma_k\) | 本次动作更新的整体幅度 | 有界标量 |

动作更新写为：

\[
A_{k+1}
=
A_k+s_k\gamma_k\Delta A_k
\]

其中 \(\Delta A_k\) 由带有第 \(k\) 组 gated-LoRA 的 MMDiT refine block 产生。

### 2.3 推理时保持 LoRA 未合并

如果 \(g_k\) 随样本、条件或 refine step 动态变化，则：

\[
\Delta W_k(x)=U_k\operatorname{diag}(g_k(x))V_k
\]

无法提前静态合并进 \(W_0\)。因此推理阶段应：

- 将共享 MMDiT 和所有 step-specific LoRA 常驻设备；
- 每个 block 直接调用自己的 LoRA 参数；
- 不在 refine step 之间执行运行时 merge/unmerge；
- 保留 gate、rank 和 residual scale 的运行时记录。

该选择的目的不是节省计算，而是保留动态参数结构、逐步可解释性和消融能力。

---

## 3. 总体网络结构

### 3.1 初步结构

\[
A_0 = \operatorname{InitialMMDiT}(L,S,X_t,t)
\]

\[
\begin{aligned}
A_1 &= A_0+s_1\gamma_1R_1(A_0;C_1,\Delta\theta_1) \\
A_2 &= A_1+s_2\gamma_2R_2(A_1;C_2,\Delta\theta_2) \\
A_3 &= A_2+s_3\gamma_3R_3(A_2;C_3,\Delta\theta_3)
\end{aligned}
\]

### 3.2 共享部分

以下能力由共享 MMDiT 提供：

- 基础 multimodal token interaction；
- 通用时间编码与 noisy-action 理解；
- 通用 action representation；
- 公共 attention/FFN 变换能力；
- 共享的 arm/gripper 表达底座。

### 3.3 每个 refine block 的独立部分

每个 \(R_k\) 独立拥有：

- 明确的 condition projection；
- LoRA \(U_k,V_k\)；
- rank gate；
- block gate；
- residual scale；
- 输出统计和诊断接口。

三个 block 在代码中显式构造，不通过循环式模块 loader 动态装配。

---

## 4. 条件注入设计

### 4.1 条件权限原则

1. 条件流保持 read-only；
2. Action token 只能被 refine block 更新，不能写回 Workspace；
3. gate 只能控制参数修改子空间和输出采用程度；
4. 不允许 gate 直接改写 evidence value；
5. 每个 block 的 forward signature 明确列出允许读取的条件；
6. 不把所有条件塞入统一字典后交给通用 condition loader。

### 4.2 第一轮实验的条件分配假设

下表是第一轮可验证假设，不作为未经实验的最终语义结论：

| Block | 初始职责假设 | 允许读取的主要条件 | 预期更新性质 |
| --- | --- | --- | --- |
| Refine 1 | 底层证据对齐与大方向修正 | \(A_0,L_k,X_t,t\) | 较宽、较明显的动作修正 |
| Refine 2 | 阶段一致性与进度协调 | \(A_1,S_k,progress,t\) | 阶段性、轨迹级修正 |
| Refine 3 | 局部精度与末端修正 | \(A_2,micro,gripper/terminal,t\) | 较窄、较局部的最终修正 |

如果现有源码语义表明某个条件不属于对应层级，应根据真实信息流调整，不为叙事强行制造职责。

### 4.3 条件注入位置的首轮选择

首轮优先在以下位置注入 step-specific LoRA：

1. condition-to-action 的 value/output projection；
2. action stream FFN；
3. action stream 的调制层或 residual projection。

首轮暂不让 LoRA 大范围修改所有 Q/K 投影，原因是 Q/K 决定读取路径，过早开放可能使各 block 通过注意力路由重新混合职责。若条件消融表明读取能力不足，再单独开放指定 Q/K 投影。

---

## 5. Gated-LoRA 具体参数化

### 5.1 基本形式

对输入 \(x\)：

\[
\Delta y_k
=
U_k\left[g_k(C_k)\odot(V_kx)\right]
\]

其中：

- \(V_kx\)：将输入投影到低秩修改空间；
- \(g_k(C_k)\)：依据当前条件控制各个修改方向；
- \(U_k\)：将受控的低秩更新写回原隐藏空间。

### 5.2 Gate 粒度

第一阶段使用 rank group，而不是逐个 rank channel：

```text
Group 1: rank channels  1-4
Group 2: rank channels  5-8
Group 3: rank channels  9-16
Group 4: rank channels 17-32
```

原因：

- 单个 rank channel 的基可以旋转，单通道语义不稳定；
- group 更容易形成稳定统计；
- group-level 有效 rank 更容易解释和消融；
- 避免 gate 维度过高后成为新的隐藏黑盒。

### 5.3 Rank 初始设置

首轮不强制 \(r_1>r_2>r_3\)，只将其作为待验证假设。建议从相同最大 rank 开始，例如：

```text
Refine 1: max rank = 32
Refine 2: max rank = 32
Refine 3: max rank = 32
```

训练后根据有效 rank、更新谱和条件消融结果决定是否固化为不同容量。这样可以避免先验地把“后期修正”错误等同于“更低 rank”。

### 5.4 修改幅度约束

为避免低 rank 产生过大的动作修改，对 \(\gamma_k\) 使用有界参数化：

\[
\gamma_k=\gamma_{\max,k}\cdot\sigma(\hat\gamma_k)
\]

必要时加入软 trust-region：

\[
\mathcal L_{trust}
=
\sum_k
\left[
\|\Delta A_k\|_2-\tau_k
\right]_+^2
\]

该项只惩罚超过允许范围的更新，不强迫所有 refine 残差趋近于零。

---

## 6. 训练计划

### 阶段 0：建立不可绕过的基线

任务：

- 固定当前最佳 MMDiT checkpoint；
- 记录无分块 gated-LoRA 时的训练、验证和 rollout 指标；
- 保存每个现有 refine step 的 \(A_k\) 与 \(\Delta A_k\)；
- 分别统计 arm 与 gripper field 的残差；
- 计算不同 step 的有效更新秩和奇异值能量曲线。

输出：

- `baseline_refine_metrics.json`；
- `refine_delta_spectrum.json`；
- 每个 step 的更新范数、余弦相似度和 singular energy 图表。

退出条件：

- 能够确认当前不同 refine step 是否已经形成不同更新分布；
- 能够确定合理的 LoRA 最大 rank 搜索范围。

### 阶段 1：建立显式 refine blocks，不启用动态稀疏

任务：

- 将 Refine 1、2、3 显式展开；
- 接入共享 MMDiT core；
- 为每个 block 创建独立 LoRA 和 condition projection；
- 所有 rank gate 暂时固定为全开；
- 所有 block gate 暂时固定为启用；
- 保留原主损失和 Parseval/null-space 约束。

训练策略：

- 优先冻结共享 core；
- 只训练 block-specific condition projection、LoRA、\(\gamma_k\)；
- 若能力明显不足，再对 shared core 使用远小于 LoRA 的学习率。

退出条件：

- 分块模型达到接近基线的 final flow 与 rollout 水平；
- 三个 block 均产生非零且稳定的 \(\Delta A_k\)；
- 不出现某一个 block 独占全部修正的明显塌缩。

### 阶段 2：启用 gated rank

任务：

- 将每个 LoRA rank 划分为固定 groups；
- condition router 输出 group gate；
- gate 从 soft sigmoid 开始训练；
- 稳定后再评估 hard-concrete、Gumbel 或阈值化；
- 增加有效 rank 统计。

建议损失：

\[
\mathcal L
=
\mathcal L_{main}
+\lambda_{null}\mathcal L_{null}
+\lambda_{rank}\mathcal L_{rank}
+\lambda_{trust}\mathcal L_{trust}
\]

其中：

\[
\mathcal L_{rank}
=
\sum_{k,g}w_g\left|g_{k,g}\right|
\]

注意：\(\mathcal L_{rank}\) 的目标是限制修改自由度，而不是追求最低计算量。其权重应由验证性能和可解释性共同决定，不能持续增大直至 gate 全部关闭。

退出条件：

- gate 不全开也不全关；
- 不同 block 或不同样本形成可重复的有效 rank 分布；
- rank 下降没有导致 gripper 或 tail 指标异常恶化。

### 阶段 3：启用 block gate 与 residual scale

任务：

- 训练 \(s_k\) 判断该次 refine 是否应被采用；
- 训练有界 \(\gamma_k\) 控制修改幅度；
- 记录 gate 与每步 loss gain 的对应关系；
- 对不确定或分布外样本保留较保守的 block 启用策略。

注意：

- block gate 不得读取 posterior target；
- 不让训练期可见、推理期不可见的信息参与 gate；
- gate 可以读取 Stage Memory、progress、time、refine index；
- 如需读取动作变化，只允许使用 stop-gradient 后的低维统计量，例如上一轮 \(\|\Delta A\|\)。

退出条件：

- 被关闭的 block 在对应样本上原本就只有较低增益；
- gate 与任务难度或阶段需求形成稳定关系；
- 不出现仅凭任务 ID 机械路由、对未知样本全部低估的现象。

### 阶段 4：小学习率联合收敛

任务：

- 解冻 shared core 的少量目标层；
- shared core 使用显著低于 LoRA 的学习率；
- 继续保持 rank 与 trust-region 约束；
- 检查 shared core 是否重新吸收 step-specific 能力。

退出条件：

- 联合微调确实改善 final/rollout 指标；
- 各 block 的条件依赖和 LoRA 差异没有消失；
- 参数职责仍能通过消融复现。

---

## 7. 损失与指标要求

### 7.1 保留的任务指标

- 主损失继续按 \((6\text{ arm}+1\text{ gripper field})/7\) 聚合；
- 保留 arm/gripper 分项指标；
- 保留 first、first4、first8、tail、rollout 等现有指标；
- 保留 Parseval field 的 null-space 惩罚；
- 保留 flow 与最终动作解码的一致性检查。

### 7.2 新增 block-level 指标

每个 refine block 至少记录：

| 指标 | 含义 |
| --- | --- |
| `refine_k_delta_norm` | 本步动作更新幅度 |
| `refine_k_arm_delta_norm` | arm 更新幅度 |
| `refine_k_gripper_delta_norm` | gripper field 更新幅度 |
| `refine_k_loss_gain` | 使用本 block 前后的目标误差变化 |
| `refine_k_block_gate` | block 是否被采用 |
| `refine_k_residual_scale` | \(\gamma_k\) |
| `refine_k_effective_rank` | 当前样本实际激活的 rank |
| `refine_k_rank_group_usage` | 各 rank group 的激活频率 |
| `refine_k_null_energy` | 本步引入的非法场能量 |

### 7.3 LoRA 参数指标

- 每个 block 的 \(\|U_kV_k\|_F\)；
- 每个 block 的谱范数估计；
- LoRA singular value 分布；
- 不同 block LoRA 子空间的 principal angles；
- 不同 block 输出残差的余弦相似度；
- shared core 与 LoRA 的梯度比例。

---

## 8. 可解释性与消融矩阵

### 8.1 必做消融

| 编号 | 实验 | 目的 |
| --- | --- | --- |
| A0 | Shared MMDiT，无 step-specific LoRA | 公共能力基线 |
| A1 | 三个独立 LoRA，gate 全开 | 验证分步参数化价值 |
| A2 | 单一共享 LoRA，三个 block 共用 | 验证 step-specific 参数是否必要 |
| A3 | 独立 LoRA + scalar block gate | 验证是否需要选择性 refine |
| A4 | 独立 gated-LoRA | 验证有效 rank 控制 |
| A5 | gated-LoRA，无 residual scale | 验证 rank 与幅度控制不可互相替代 |
| A6 | gated-LoRA，随机交换条件入口 | 验证 block 条件语义是否真实 |
| A7 | 每次删除一个 refine block | 测量逐块真实贡献 |
| A8 | 每次屏蔽一个 block 的特定条件 | 条件归因 |
| A9 | 三个完全独立的完整 MMDiT block | 作为自由度上界，不作为首选结构 |

### 8.2 防止伪可解释性的检查

以下情况说明结构虽然形式分块，但没有形成真实职责：

- 三个 LoRA 子空间高度重合；
- 三个 block 输出的 \(\Delta A_k\) 方向几乎相同；
- 交换条件入口后性能不变；
- 删除任意一个 block 都由后续 block 完全补偿；
- gate 始终全开、全关或只根据 refine index 固定输出；
- shared core 吸收全部能力，LoRA 参数接近零；
- 每个 block 都试图直接给出最终动作，而不是形成渐进修正。

---

## 9. 风险与应对方案

### 风险 1：Shared core 吸收全部能力

表现：LoRA norm 与 gate 长期接近零。

应对：

- 先冻结 shared core；
- LoRA 学稳后再小学习率解冻；
- 监控 shared/LoRA 梯度比例；
- 避免从头完全联合训练。

### 风险 2：LoRA 承担全部能力

表现：共享 core 退化，不同 block 无法稳定复用基础能力。

应对：

- 从已训练 shared checkpoint 开始；
- 限制 LoRA rank 和 residual scale；
- 保留 shared-only 路径的周期性验证。

### 风险 3：低 rank 但修改幅度过大

表现：有效 rank 很低，但动作跳变明显、rollout 变差。

应对：

- rank gate 与 \(\gamma_k\) 分离；
- 加入有界 residual scale；
- 使用只惩罚超限的 trust-region；
- 单独检查 gripper event 与 tail。

### 风险 4：Gate 塌缩

表现：所有 rank group 全开或全关。

应对：

- gated rank 延后启用；
- gate warm-up；
- 设置合理的目标 rank 区间，而不是无限追求稀疏；
- 使用 group gate，降低路由难度。

### 风险 5：人为分配的 block 语义不成立

表现：条件交换和条件屏蔽对结果影响很小。

应对：

- 把职责表视为实验假设；
- 用条件消融决定最终条件分配；
- 必要时合并无独立意义的 block；
- 不为论文叙事保留没有真实贡献的模块。

### 风险 6：Gripper 被低秩约束压制

表现：总体 MSE 稳定，但 gripper、event 或 tail 明显恶化。

应对：

- 单独记录 gripper effective rank 与残差幅度；
- 为最终 block 保留最低 gripper rank group；
- 保留现有 \((6+1)/7\) 聚合语义；
- 检查 Parseval null-space 能量，而不只看物理解码误差。

---

## 10. 建议代码组织

在不隐藏网络结构的前提下，建议使用少量、具有真实语义的文件：

```text
policy/
├── mmdit/
│   ├── shared_mmdit_core.py
│   └── gated_lora_refine_stack.py
├── condition/
│   └── refine_condition_interfaces.py
├── diagnostics/
│   └── refine_block_diagnostics.py
└── training/
    └── refine_regularization.py
```

要求：

- `gated_lora_refine_stack.py` 内显式定义并构造 Refine 1、2、3；
- 不使用按字符串名称动态创建 block 的 loader；
- 每个 block 的条件参数直接出现在 forward signature；
- 工具性 gate、LoRA projection 可以共用实现；
- 三个 block 的条件接口和残差路径必须在主网络代码中可直接读出。

如果三个 block 的内部条件逻辑后来各自增长为稳定的 100–300 行语义模块，再拆成独立文件；在此之前不创建只有几行封装的小文件。

---

## 11. 实施里程碑

### M0：语义与秩探针

- 导出当前各 refine step 的更新；
- 完成有效秩、残差方向和 arm/gripper 分项分析；
- 确定最大 rank 搜索范围。

**交付物：** probe 报告、谱图、初始 rank 建议。

### M1：显式三块 + 固定 LoRA

- 显式展开 Refine 1–3；
- 接入独立条件接口和 LoRA；
- gate 全开；
- 恢复至基线性能。

**交付物：** 可训练 checkpoint、逐块日志、基础消融。

### M2：动态 rank gate

- 启用 rank groups；
- 加入有效 rank 约束；
- 验证 block 间 rank 分布和职责差异。

**交付物：** gated-LoRA checkpoint、rank 使用报告。

### M3：Block gate 与幅度控制

- 加入 \(s_k\) 和 \(\gamma_k\)；
- 验证 gate 与真实 refine gain 的一致性；
- 完成未知样本 fallback 检查。

**交付物：** 完整动态 refine checkpoint、gate 校准报告。

### M4：联合微调与完整消融

- 小学习率解冻 shared core；
- 完成 A0–A9 消融；
- 固化最终条件分配；
- 形成论文结构描述与网络图语义。

**交付物：** 最终 checkpoint、消融表、结构图输入说明。

---

## 12. 最终验收标准

方案只有同时满足以下条件才算成立：

1. 每个 refine block 对最终输出具有可重复、可测量的独立贡献；
2. 不同 block 对条件屏蔽呈现不同敏感性；
3. step-specific LoRA 优于或明显不同于单一共享 LoRA；
4. 有效 rank 分布不是全开、全关或完全固定；
5. rank 与 residual scale 对行为具有可分离影响；
6. shared core 与 step-specific LoRA 没有相互吞并；
7. arm、gripper、tail、rollout 和 null-space 指标均保持健康；
8. Action 不写回 Workspace，证据读取边界保持不变；
9. 推理时未合并 gated-LoRA 能稳定复现训练期行为；
10. 删除、交换或屏蔽 block/condition 后的性能变化能够支持其结构叙事。

---

## 13. 推荐的第一步

暂不直接重写训练主干。先完成阶段 M0：

1. 在现有 checkpoint 上导出 \(A_0,A_1,A_2,A_3\)；
2. 计算每步 \(\Delta A_k\) 的 singular energy；
3. 分别统计 arm 与 gripper field 的有效秩；
4. 计算三个 step 更新方向的相似度；
5. 检查当前 refine 是否已经自然形成“广域修正—阶段修正—局部修正”。

如果当前更新确实呈现不同结构，再进入显式三块改造；如果三个 step 本质相同，则应先解决 refine 职责未分化的问题，而不是立即给三个相同过程添加 LoRA 名称。
