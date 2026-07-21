> 命名台账: 本文件原名 do_before_v76.md。其实施产物即 v76a/v77 运行世代（B 系列+CR 系列）。代码注释中引用的 "do_before_v76 §N" 均指本文件章节。

# CVAE 残余清算与确定性意图合同取代计划

> 工作名称：**CVAE Replacement Program**
>
> 核心原则：不是把 CVAE 删除后留下空位，而是先识别它目前承担的全部实际职责，再用职责更明确的新组件逐项接管；只有完成接管后，才清理旧模块、损失、接口和命名。
>
> 本计划应先于 `Exhaustion-Driven Stage-Adaptive Refinement` 的正式落地。否则自适应 refinement 会继续建立在 `z-primary` 条件总线上，后续迁移成本会进一步扩大。

---

# 1. 项目目标

当前 V75 的 CVAE 已经不再只是一个独立 latent head。它经过 V60、V65 到 V75 的演化，逐渐成为：

1. 全局条件压缩器；
2. Workspace 的初始化与查询条件；
3. MMDiT 的主调制条件；
4. 输出 transition/event head 的直接条件；
5. 旧 CVAE action stem 的驱动条件；
6. 训练阶段 target-conditioned posterior 分支；
7. 一整套 KL、posterior reconstruction、指标、CLI 和 checkpoint 合同。

因此本项目分成两个部分。

## Part A：清算 CVAE 时代残余

目标是识别所有仍在 active forward、训练目标、参数图、配置、日志和 checkpoint 中产生影响的 CVAE 遗留，并逐项消除其不应存在的作用。

## Part B：用确定性意图合同取代 CVAE

目标是构建一个：

- 确定性；
- 无 prior/posterior；
- 无 Gaussian 假设；
- 无 KL；
- 无 target-conditioned 推理辅助路径；
- 无额外训练损失；
- 职责显式分离；
- 不形成新的中央捷径；

的 **Intent Contract Compiler（ICC）**，完整接管 CVAE 当前合理但混杂的职责。

---

# 2. 当前结构的历史依赖链

代码历史显示：

```text
V60
MMDiT-lite 接管动作更新
CVAE prior/posterior 仍保留

V65
z 被提升为 MMDiT / Workspace 的 primary condition

V75
Hierarchical Workspace 接入同一条 z-primary 控制总线
```

所以当前不是：

```text
CVAE head → 一个普通 condition token
```

而是：

```text
Condition Organizer
        ↓
CVAE prior mean z
        ↓
primary_cond = z_to_token(z) + diffusion_time
        ├─→ Stage Memory 初始化
        ├─→ Workspace selector / manager
        ├─→ Stage promotion
        ├─→ MMDiT AdaLN / modulation
        ├─→ Physical output transition head
        └─→ Legacy CVAE action stem
```

这意味着 CVAE 已经变成中央控制合同。取代工作必须沿整个依赖链进行，不能只替换 `prior`。

---

# Part A：CVAE 残余清算

# 3. A1：Variational 残余

当前部署主路径使用：

\[
z=\mu_p(c)
\]

因为 `latent_cvae_inference_sample=0`，推理时不采样。

但训练仍保留：

- prior `μ/logvar`；
- posterior `μ/logvar`；
- posterior action encoder；
- Gaussian sampling；
- KL divergence；
- posterior full decode；
- posterior flow reconstruction；
- posterior decoded-action loss；
- posterior statistics 和梯度日志。

## 问题

### 3.1 部署语义与训练机制不一致

部署需要的是一个确定性的意图编码：

```text
condition → intent
```

训练却仍把它定义为两个 Gaussian 分布的匹配问题：

```text
p(z|condition)
q(z|condition,target_action)
```

这会把表示空间约束为“适合 prior/posterior Gaussian 对齐”，而不一定是“最适合驱动 MMDiT 和 Workspace”。

### 3.2 Posterior 分支重复执行完整 decoder

当 target 存在时，代码不仅计算 posterior latent，还再次执行：

```text
Hierarchical Workspace
MMDiT refinement
Physical output head
```

这会增加训练计算、显存、日志和梯度路径复杂度。

### 3.3 Target action 深入 policy API

`cvae_target_physical` 从 runtime 进入 policy，再进入 posterior。目标动作不再只是 loss target，而成为模型内部训练分支的输入。

这会形成一套和部署主线不同的内部合同，并增加未来代码误用或 target leakage 的风险。

### 3.4 Posterior 对整段 action 做 mean pooling

当前 posterior action encoder 最终按 horizon mean 得到全局 target feature。

它会鼓励 latent 捕获：

- 整体动作方向；
- task/action template；
- 平均 gripper 状态；

但弱化精确时序。

如果我们真正需要的是确定性意图，这种 target-global-template 塑形并不是必要机制。

### 3.5 `mu_bound` 与 `min_std` 属于无意义遗留约束

当前还有：

- latent mean tanh bound；
- minimum standard deviation；
- log variance clamp。

这些约束服务于 Gaussian latent，而部署使用的是 deterministic prior mean。它们可能限制控制表示的动态范围，却不提供实际采样收益。

---

# 4. A2：`z-primary` 中央总线

当前同一个 `primary_cond` 同时控制：

1. Stage Memory 初始化；
2. Workspace low selector；
3. Workspace manager；
4. Stage promotion；
5. low workspace block modulation；
6. 每个 MMDiT block modulation；
7. final event/transition head；
8. legacy action stem。

## 问题

### 4.1 角色冲突

同一个向量同时要表达：

```text
任务意图
阶段初始状态
应读取什么证据
动作更新动力学
输出 transition 偏置
```

这些职责并不等价。

来自不同模块的梯度都会回流到同一个 latent：

\[
\nabla_z
=
\nabla_z^{workspace}
+
\nabla_z^{stage}
+
\nabla_z^{mmdit}
+
\nabla_z^{output}
+
\nabla_z^{legacy}
\]

模型可能找到一个对多数分支都“勉强可用”的混合表示，而不是对每个职责都清晰的表示。

### 4.2 中央捷径

如果 \(z\) 已经能编码 task identity 或平均动作模板，那么：

```text
z → Stage / MMDiT / Output
```

可能绕开对细粒度 EvidenceBank 的依赖。

### 4.3 难以归因

当性能变化时，无法区分：

- CVAE latent 是否更好；
- Workspace 是否更好；
- MMDiT 是否更好；
- output head 是否直接利用了 z；
- legacy stem 是否已经预生成了动作偏置。

---

# 5. A3：意图与 diffusion time 过早纠缠

当前：

\[
primary\_cond
=
LN(W_z z+W_t e(t))
\]

然后这个混合向量被用于：

- Stage Memory 初始化；
- Workspace selection；
- MMDiT modulation；
- output head。

## 问题

意图是一次 action-flow evaluation 内应相对稳定的语义：

```text
要做什么
希望达到什么局部效果
当前动作属于哪种控制上下文
```

Diffusion time 则表示：

```text
当前 noisy action 所处的生成阶段
```

两者过早混合会导致：

- Stage Memory 初始状态随 flow time 改变；
- Workspace 读取策略随噪声时间改变；
- 意图表示无法独立诊断；
- 后续自适应 refinement 难以区分“任务阶段”和“去噪阶段”。

## 清算原则

新结构中必须分离：

```text
time-invariant intent contract
diffusion-time modulation
refinement-stage identity
remaining computation budget
```

它们只在真正需要的模块入口处组合。

---

# 6. A4：条件信息重复拥有

同一批语义目前同时进入两条路径。

## 全局路径

```text
layer / transition / trajectory / scan / lateral
        ↓
condition
        ↓
CVAE z
        ↓
primary_cond
```

## 细节路径

```text
layer / transition / trajectory / scan / lateral / rollout
        ↓
EvidenceBank
        ↓
L_k / S_k
```

## 问题

### 6.1 同源信息重复注入

MMDiT 同时接收：

- 由同一证据压缩出的 global latent；
- 原始或细粒度 evidence read。

它可能只使用更短、更容易的 global path。

### 6.2 Source ownership 不清楚

目前无法明确回答：

```text
scan/lateral 应属于 global intent，还是 workspace evidence？
layer summary 应用于 intent，还是低层读取？
trajectory summary 与 full trajectory 的边界是什么？
transition delta 应决定意图，还是作为每步证据？
```

### 6.3 两条路径可能冲突

global latent 是一次性压缩结果，Workspace 是逐步读取结果。两者可能对同一情景产生不同偏置，迫使 MMDiT 学习抵消。

## 清算原则

必须建立 source ownership matrix：

```text
高层摘要 → Intent Contract
高分辨率事实 → EvidenceBank
flow noisy action → MMDiT noisy group
当前 action → action stream
```

同一种语义默认只能有一个主所有者。重复路径只能作为明确消融，而不能长期隐式并存。

---

# 7. A5：冻结的 legacy CVAE action stem

当前 hierarchical 主线仍执行：

```python
for block in self.blocks:
    action = block(action, cond_time)
```

其中 `self.blocks` 是三个 `LatentCVAEActionBlock`。

代码对其调用：

```python
self.blocks.requires_grad_(False)
```

但冻结参数不等于旁路 forward。

## 为什么它仍会改变 action tokens

每个 block 仍执行：

\[
A'
=
A+
g_{attn}(c)\,\mathrm{SelfAttn}(A)
\]

\[
A''
=
A'+
g_{ffn}(c)\,\mathrm{FFN}(A')
\]

只要 checkpoint 中 modulation gate 非零：

\[
A''\ne A
\]

此外：

- 梯度仍可穿过固定 block 回到 `primary_cond`；
- train mode 下 Dropout 仍可能生效；
- 如果加载过旧训练权重，它会成为一个固定但活跃的条件算子；
- 如果从零初始化即冻结，它可能近似 identity，但必须测量而不能假设。

## 风险

```text
Horizon seed
    ↓
z-conditioned legacy transformation
    ↓
already biased action state
    ↓
Hierarchical MMDiT
```

MMDiT 的前几步可能用于纠正旧 stem，而不是解决当前 evidence-action refinement。

## 必须新增的探针

\[
r_{stem}
=
\frac{\|A_{after}-A_{before}\|_F}
{\|A_{before}\|_F+\epsilon}
\]

同时记录：

- per-sample stem effect；
- stem on/off 后的 final RMSE；
- stem on/off 后每个 refinement step 的 update norm；
- train/eval 模式差异；
- stem 对 `primary_cond` 的梯度敏感性。

---

# 8. A6：Output head 的直接条件捷径

当前 `_emit_action(action, cond)` 将 `cond` 扩展到 horizon 后与 action 拼接：

```text
[action token, primary_cond]
        ↓
event gate
event transition
        ↓
physical velocity head
```

因此 `primary_cond` 可以在 MMDiT 之后再次直接影响输出。

## 风险

即使 MMDiT 没有充分利用 evidence，output head 也可能通过 global condition 恢复一部分 task/action template。

## 清算原则

目标主线第一版应采用：

```text
final action tokens
    ↓
physical output head
```

如果 event/transition 确实需要额外条件，应使用一个独立、受限的 `output contract`，不能复用整个 global intent bus。

该路径必须单独消融：

```text
action-only output
action + dedicated output contract
action + old primary_cond
```

---

# 9. A7：hierarchical 分支中的冗余计算与伪接口

在 hierarchical MMDiT 分支中，真正的 condition token group 是：

```text
low evidence
stage memory
noisy action
```

但代码仍然计算和传递：

- `z_token`；
- legacy condition-token arguments；
- 部分 trajectory/rollout/global projection；
- progress 相关接口。

这些参数在该分支可能不进入最终 token layout，却仍保留在 API 和模块构造中。

## 风险

- 误以为 z 仍通过 condition attention 注入；
- 无效参数长期留在 state dict；
- 分支修改时意外重新激活；
- 图、README 和代码语义不一致。

---

# 10. A8：继承体系带来的 inactive trainable modules

`AdaptiveRecurrentCVAEActionDecoder` 继承 `LatentCVAEActionDecoder`，并继续构造大量历史模块：

```text
legacy progress memory
context capsule blocks
route projections
function banks
micro controller
micro refine block
legacy refine block
semantic adapters
output adapters
legacy workspace
posterior modules
```

V75 主线关闭了许多功能，但模块仍可能：

- 被实例化；
- 出现在 state dict；
- 出现在 DDP 参数图；
- 增加 checkpoint 体积；
- 增加模型初始化和加载复杂度；
- 造成 unused parameter 处理成本；
- 在未来 flag 组合下被意外重新启用。

## 清算原则

最终必须建立全新的：

```python
HierarchicalMMDiTActionDecoder
```

不得继承 CVAE decoder 或 adaptive CVAE decoder。

---

# 11. A9：配置、日志与训练 API 残余

当前代码中存在大量：

```text
latent_cvae_*
adaptive_cvae_*
cvae_target_physical
post_*
cpz / ckl / cpflow
```

残余范围包括：

- policy config；
- CLI parser；
- shell scripts；
- runtime losses；
- optimizer groups；
- gradient metrics；
- progress logs；
- checkpoint keys；
- README；
- class names；
- output dictionary。

当前代码中可检索到约：

```text
101 种 latent-cvae CLI 名称
58 种 adaptive-cvae CLI 名称
```

并非所有都 active，但它们会持续污染主线语义。

## 清算原则

旧 checkpoint 兼容只能存在于：

```text
legacy checkpoint migration loader
```

不能继续存在于新 active decoder 的命名和配置中。

---

# 12. A10：多模态职责重复

当前系统同时具备：

```text
CVAE latent stochasticity
action flow initial noise
```

但部署又使用 deterministic prior mean。

因此 CVAE 并没有真正承担部署时的多模态采样；动作多模态本来就可以由 conditional diffusion/flow 生成过程承担。

Diffusion Policy 已明确验证条件动作扩散能够处理多模态动作分布；因此新结构中应让 flow noise 成为唯一的生成随机源，而意图合同保持确定性。

---

# 13. A11：Absolute step 与条件遗产的耦合

Hierarchical Workspace 当前仍有 absolute step embedding。

未来的自适应 refinement 要求：

```text
block stage ≠ absolute step
```

如果 condition replacement 完成后仍保留绝对 step 作为强条件，模型可能继续学习固定路径。

该问题不属于 CVAE 本身，但必须在本次结构清算中标记，并在自适应 refinement 落地前改成：

```text
stage identity
+
remaining budget
```

---

# 13.5. A12：审计追加项（登记与现状）

| 项 | 现状 |
|---|---|
| WorkspaceController 的 action→workspace 值 FiLM（v74b 红线） | **已修**：value_state（无动作）喂 workspace_mod，select_state（含动作）只喂选择级控制；防火墙契约已写入类 docstring |
| 旧 bank "event" 幽灵角色（wevt 在 B0/B1 恒零） | **已登记**：ROLE_NAMES 处代码注释 + B1 脚本判读手册注记；正式清理归 CR9 |
| CR0 探针 | r_stem 无条件仪表已上（rstem=）；z zero/shuffle 干预探针 flag 门控已上（latent_cvae_z_probe，诊断跑专用） |
| CR7 g_out 备选 | 已实现（hierarchical_mmdit_output_contract，零初始化、只进 event/motion 子头） |
| CR1/B1 判决臂 | 已实现（latent_cvae_variational=0，部署映射逐位不变） |
| **法证修正：乘法 t-gate 自 v54 起在所有 MMDiT block 内部无效** | cond_norm 为尺度不变 LayerNorm，LN(g·x)=LN(x)，外置门的键与值缩放均被抵消。v70 案回溯改判：当年真实变量是 log g(t) 的 logit 偏置（score 域不被 LN 抵消），非值音量；"模型自选安静 x_t"（mdnt 读数）叙事作废——mdnt 量的是块外范数 |
| v76a E1 x_t 饿死（hmdna→0.4%，pfnu 钉 1.0 地板） | 根因=从零初始化的注意力赢者通吃锁死 + low/stage 有可学习 logit 补贴而 noisy 无对等杠杆。修复：hierarchical_mmdit_noisy_market_bias（零初始化可学习组偏置，梯度不随份额塌缩饱和，默认开）；hierarchical_mmdit_noisy_gate_mode（0=诚实无门=历史全部胜者的实际机制，默认；1=块内 post-norm 值域真门，留作低 t 泄漏防护的可测试项） |

---

# Part A 实施计划

# 14. CR0：无行为变化的残余审计

目标：先测清楚每条遗留路径当前到底承担多少功能。

## 新增 probes

### 14.1 Legacy stem

```text
legacy_stem_effect_ratio
legacy_stem_effect_p50/p90
legacy_stem_train_eval_gap
```

### 14.2 z 依赖

```text
z_zero_delta
z_shuffle_delta
z_scale_sensitivity
primary_z_effect_per_sample
```

### 14.3 路径级 intervention

分别只允许 global condition 进入：

```text
Workspace only
MMDiT modulation only
Output head only
Legacy stem only
```

测量每条路径的独立贡献。

### 14.4 Evidence reliance

```text
global sources off
layer source off
trajectory source off
rollout source off
transition source off
```

检查 z-primary 是否已经压制 EvidenceBank。

### 14.5 Posterior 成本

记录：

```text
posterior branch wall time
posterior branch activation memory
posterior branch gradient norm
posterior contribution to total loss
```

### 14.6 Active parameter audit

用 forward hooks / grad hooks 输出：

```text
parameter group
forward-called
received-gradient
state-dict-only
```

形成 active / inactive module 清单。

## CR0 输出

1. `cvae_residual_dependency_report.json`
2. `cvae_residual_dependency_report.md`
3. 每条路径的 effect table
4. state dict / parameter ownership map
5. 新旧结构迁移风险排序

---

# 15. CR1：Variational 机制旁路，保持部署函数不变

目标：判断当前收益是否来自 variational training，还是仅来自 deterministic prior-mean mapping。

## 做法

建立：

```python
LegacyDeterministicControlAdapter
```

它严格复用当前部署映射：

\[
g_{legacy}(c)
=
W_z\,\mu_p(c)
\]

但不再构造：

- p_logvar；
- posterior；
- sampling；
- KL；
- posterior full decode；
- posterior reconstruction。

该阶段暂时不改变：

- Workspace；
- MMDiT；
- legacy stem；
- output head。

## 对照

```text
CR1-A：原 V75 CVAE
CR1-B：相同 p_mu 映射，但无 posterior/KL
```

## 目的

回答唯一问题：

> Variational 训练机制本身是否贡献了可复现的性能收益？

如果 CR1-B 与 V75 相当，则可以确认：

```text
CVAE 当前真正有用的是 deterministic condition transform
而不是 variational latent
```

---

# 16. CR2：意图、时间、阶段合同解耦

目标：不再使用单一 `primary_cond` 作为中央总线。

先在保持旧 latent mapping 的情况下，将接口拆成：

```text
intent control
stage seed
evidence read anchor
diffusion time
refinement stage
```

即使底层暂时仍由旧 p_mu 提供，也先改变接口结构。

这样可以区分：

- 表示替代风险；
- 接口重构风险。

---

# Part B：确定性意图合同取代方案

# 17. 设计判断

CVAE 当时真正承担的合理意图是：

> 给 MMDiT 一个比原始 condition 更好的、动作相关的全局条件编码。

但现在：

- action flow 已负责生成分布和多模态；
- Condition Organizer 已负责跨层压缩；
- EvidenceBank 已保留细节事实；
- Stage Memory 已负责 refinement 阶段状态。

因此新的组件不需要再是 latent generative model。

它应该是：

> **将已组织条件编译为多个职责明确、确定性的控制合同。**

---

# 18. 首选方案：Intent Contract Compiler（ICC）

## 18.1 为什么不直接用另一个 latent

不建议：

```text
condition → another hidden z → everything
```

因为这只会重新制造一个中央混合瓶颈。

## 18.2 为什么首版不增加 query attention

Perceiver、Set Transformer PMA 和 BLIP-2 Q-Former 都证明了 learnable queries 可以从大量输入 token 中提取固定数量表示。

但当前项目已经有：

- GRU layer scan；
- lateral condition fusion；
- typed EvidenceBank；
- hierarchical workspace。

再加一个 query-attention 汇聚器，可能产生另一个松散的 pseudo-workspace。

因此首版 ICC 使用：

```text
现有高层条件摘要
+
显式角色投影
```

不增加新的 attention block，也不增加辅助 loss。

---

# 19. ICC 输入所有权

ICC 只读取高层、稳定、与 intent 有关的摘要：

```text
c_scan
c_lateral
transition summary
trajectory summary
task/state global summary
```

ICC 默认不读取：

```text
noisy action x_t
current action A_k
full rollout grid
full trajectory tokens
low-level visual tokens
target action
stage memory
absolute refine step
```

这些信息分别属于：

```text
x_t              → noisy-action group
A_k              → action stream
full evidence    → EvidenceBank
S_k              → Stage Memory
target action    → loss only
```

这样可以限制 intent contract 变成动作模板捷径。

---

# 20. ICC 核心公式

先形成确定性的高层语义合同输入：

\[
C
=
\operatorname{LN}
\left(
[
c_{\text{scan}},
\alpha c_{\text{lateral}},
c_{\text{transition}},
c_{\text{trajectory}},
c_{\text{global}}
]
\right)
\]

再由彼此独立的角色投影产生三个合同：

\[
\boxed{
g=P_g(C),\qquad
S_0=\operatorname{reshape}(P_s(C)),\qquad
Q_0=\operatorname{reshape}(P_q(C))
}
\]

其中：

- \(g\)：Global Intent Control；
- \(S_0\)：Stage Memory Initialization；
- \(Q_0\)：Evidence Read Anchor。

这三个投影不共享最后一层参数，避免一个混合 latent 同时承担全部职责。

---

# 21. 三个合同的职责

## 21.1 Global Intent Control \(g\)

只负责改变动作求解器的工作方式：

```text
MMDiT AdaLN / modulation
可选 flow-head global modulation
```

它不生成 EvidenceBank values，不进入 action seed。

与 diffusion time 的组合只发生在 MMDiT block 入口：

\[
m_k
=
LN\left(
W_g g
+
W_t e(t)
+
W_b e(i_k)
\right)
\]

其中：

- \(t\)：flow time；
- \(i_k\)：当前 refinement block/stage；
- 两者不写回 intent representation。

## 21.2 Stage Seed \(S_0\)

直接初始化 persistent stage slots：

\[
S_0
=
S_{\text{learned}}
+
P_s(C)
\]

它不包含 flow time。

后续只能通过 evidence promotion 和 Stage GRU 更新。

## 21.3 Evidence Read Anchor \(Q_0\)

用于初始化 low-level selector query：

\[
Q_k
=
Q_0
+
E_{\text{stage}}(i_k)
+
E_{\text{budget}}(b_k)
+
\Delta Q(S_k)
\]

它只决定读什么，不改写 raw evidence values。

---

# 22. Action State Initializer：取代 legacy CVAE stem

CVAE condition head 的取代还不够。必须用新组件接管旧 action stem 的合理职责。

## 目标

只建立：

- horizon 位置结构；
- action token 的时间组织；
- 合理的初始数值尺度。

不得提前解决 task。

## 推荐结构

```text
Learned horizon queries
        ↓
LayerNorm
        ↓
small causal temporal block
        ↓
A_0
```

约束：

```text
不读取 g
不读取 S_0
不读取 Q_0
不读取 EvidenceBank
不读取 noisy action
不读取 target action
```

首版可以直接使用：

```text
coarse_temporal_base
```

若不足，再加入一个轻量 causal self-attention 或 temporal convolution block。

它必须是 condition-neutral。

---

# 23. Output contract 的处理

首选主线：

```text
final action tokens
        ↓
physical output head
```

即 action-only output。

若 event/transition head 确实需要全局条件，则新增独立：

```text
g_out = P_out(C)
```

并限制：

- 只进入 event/transition 子头；
- 不进入 velocity 主干；
- 不与 `g` 共享最后一层；
- 必须做 action-only 对照。

不允许继续把同一个 global control 广播到所有 horizon 后直接拼接输出。

---

# 24. Query-resampler 备选方案

如果显式投影 ICC 在实验中表现为：

- 单一高层向量压缩不足；
- scan/lateral 无法保留多个并行意图维度；
- 不同合同高度相关或角色塌缩；

再升级为：

```text
Role Query Intent Compiler
```

使用少量固定角色 query：

```text
q_global
q_stage
q_read
q_output（可选）
```

对高层 typed evidence 做一次 cross-attention。

这一路线有成熟先例：

- Perceiver 使用非对称 cross-attention 将高维输入蒸馏到紧凑 latent；
- Set Transformer PMA 使用 learned seed vectors 聚合集合；
- BLIP-2 Q-Former 使用 learnable queries 提取固定数量、对下游最有用的特征。

但它只作为第二级方案，不作为首版，以避免重新引入不必要的 attention 和混合 token 系统。

---

# 25. 新结构完整数据流

```text
Layer contracts / transition / trajectory / globals
                         │
                         ▼
             Policy Condition Organizer
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
      high-level summaries    detailed evidence
                │                 │
                ▼                 ▼
      Intent Contract Compiler  EvidenceBank
       ┌────────┼────────┐          │
       │        │        │          │
       ▼        ▼        ▼          │
       g       S_0      Q_0          │
       │        │        │          │
       │        ▼        └─────┐    │
       │    Stage Memory       │    │
       │                       ▼    ▼
       │                   Low Evidence L_k
       │                       │
       └──────────────┬────────┘
                      ▼
Horizon queries → Action State Initializer → A_0
                      │
                      ▼
            Hierarchical MMDiT Refinement
                      │
                      ▼
             Physical Action Flow Head
```

多模态来源只保留：

\[
x_0\sim\mathcal N(0,I)
\]

以及 conditional action flow。

---

# Part B 实施计划

# 26. CR3：ICC Shadow Path

新增 ICC，但不控制主模型。

同时计算：

```text
old z-primary contracts
new ICC contracts
```

记录：

- norm；
- cosine；
- per-task variance；
- time invariance；
- stage/read contract correlation；
- contract 对 target error 的线性可预测性；
- contract 对 EvidenceBank source 的敏感性。

不增加 alignment loss。

目标是确认新合同数值稳定，且不同角色不会立即塌缩成同一向量。

---

# 27. CR4：ICC 接管 Workspace，旧 z 仍控制 MMDiT

分步接管，避免一次改变所有分布。

```text
Stage Seed：ICC
Read Anchor：ICC
MMDiT modulation：old z
Output head：old path
```

目标是验证 Workspace 可以脱离 CVAE latent 正常工作。

---

# 28. CR5：ICC 接管 MMDiT modulation

改为：

```text
Global Intent g + separate diffusion time + stage identity
        ↓
MMDiT modulation
```

此时 old z 只保留在 output/legacy stem 中。

对照：

```text
z-primary modulation
ICC intent modulation
raw condition modulation
```

---

# 29. CR6：Action State Initializer 接管 legacy stem

先并行实现：

```text
legacy stem
new Action State Initializer
```

进行独立 A/B，不使用 blend 作为最终论文结果。

检查：

- A0 norm 和频谱；
- horizon token diversity；
- first-step MMDiT update；
- 6-step边际改善；
- final RMSE；
- rollout；
- gripper event；
- train/eval gap。

新 initializer 接管后，legacy stem 退出 active forward。

---

# 30. CR7：Output shortcut 清理

对照：

```text
old primary_cond output head
action-only output head
dedicated output contract
```

正式主线优先选择 action-only；只有证据表明 dedicated output contract 必要时才保留。

---

# 31. CR8：建立干净的新 Decoder Class

新建：

```python
class HierarchicalMMDiTActionDecoder(nn.Module):
```

只包含：

```text
PolicyConditionOrganizer
IntentContractCompiler
ActionStateInitializer
HierarchicalEvidenceWorkspace
EvidenceActionMMDiTBlock
PhysicalActionFlowHead
```

不继承：

```text
LatentCVAEActionDecoder
AdaptiveRecurrentCVAEActionDecoder
```

新 block 改名：

```text
LatentCVAEMMDiTBlock
→ EvidenceActionMMDiTBlock
```

新 decoder 不接收 `target_physical`。

---

# 32. CR9：配置、日志与 checkpoint 迁移

## 新 active config

```text
--final-action-decoder hierarchical_mmdit_action
--intent-contract-mode explicit
--intent-contract-hidden-size
--intent-stage-slots
--intent-read-slots
--action-state-initializer
--mmdit-depth
--mmdit-refine-steps
--hierarchical-workspace
```

## Legacy config

所有：

```text
latent_cvae_*
adaptive_cvae_*
```

只保留在 legacy decoder 和 migration code 中。

## 新日志

```text
intent_global_norm
intent_stage_seed_norm
intent_read_anchor_norm
intent_contract_cosine_gs
intent_contract_cosine_gq
intent_contract_cosine_sq

action_initializer_effect
action_initializer_horizon_diversity

workspace_intent_dependence
mmdit_intent_effect
output_contract_effect
```

删除 active 主线中的：

```text
ckl
cpz
cpflow
cstd
prior/posterior metrics
```

## Checkpoint migration

提供独立工具：

```text
migrate_v75_to_intent_contract.py
```

作用：

- 加载 V75 公共 trunk；
- 加载 Workspace/MMDiT/physical head；
- 忽略 CVAE posterior；
- 可选加载 legacy p_mu adapter 作为 CR1 baseline；
- 不把 CVAE key alias 带入新 class。

---

# 33. Source Ownership Matrix

| 信息 | Intent Compiler | EvidenceBank | Stage Memory | Action Stream | Noisy Group |
|---|---:|---:|---:|---:|---:|
| Task/global summary | 主所有者 | 默认不重复 | 否 | 否 | 否 |
| Layer scan summary | 主所有者 | 默认不重复 | 否 | 否 | 否 |
| Full layer summaries | 否或仅摘要 | 主所有者 | 逐步吸收 | 否 | 否 |
| Transition summary | 主所有者 | 可选细节版本 | 逐步吸收 | 否 | 否 |
| Transition timeline | 否 | 主所有者 | 逐步吸收 | 否 | 否 |
| Trajectory summary | 主所有者 | 否 | 否 | 否 | 否 |
| Full trajectory tokens | 否 | 主所有者 | 逐步吸收 | 否 | 否 |
| Rollout grid | 否 | 主所有者 | 逐步吸收 | 否 | 否 |
| Current action \(A_k\) | 否 | 禁止 | 禁止写入 | 主所有者 | 否 |
| Noisy action \(x_t\) | 禁止 | 禁止 | 禁止 | 否 | 主所有者 |
| Target action | 禁止 | 禁止 | 禁止 | loss only | 禁止 |
| Diffusion time | 独立，不写入 intent | 可选独立调制 | 默认不初始化 | MMDiT调制 | gate |
| Absolute step | 禁止 | 禁止 | 改为 stage/budget | block stage | 否 |

---

# 34. 实验矩阵

## Baseline

### B0：当前 V75

```text
CVAE prior/posterior
z-primary
legacy stem
hierarchical workspace
```

## Variational 价值

### B1：Deterministic legacy mapping

```text
same p_mu mapping
no posterior
no KL
no posterior decode
```

## 最小确定性取代

### B2：Single deterministic control adapter

```text
condition → MLP → global control
```

用于判断一个简单 deterministic encoder 是否已经足够。

## 推荐主线

### B3：Explicit Intent Contract Compiler

```text
condition summaries
→ g / S0 / Q0
```

## Stem 清理

### B4：B3 + new Action State Initializer

## Source 清理

### B5：B4 + strict source ownership

## Output 清理

### B6：B5 + action-only output head

## 高容量备选

### B7：Role Query Intent Compiler

只有 B3 显示明显容量不足时才启用。

---

# 35. 必须完成的 probes

## 35.1 CVAE / Intent 依赖

```text
zero
shuffle
scale
batch-swap
task-group swap
```

理想结果：

- global intent 变化会改变总体动作模式；
- 不应完全恢复完整动作；
- 局部 evidence 仍决定精确执行。

## 35.2 Evidence 必要性

保留 intent，屏蔽 EvidenceBank。

若性能几乎不变：

```text
intent contract 过强，形成动作模板捷径
```

## 35.3 Intent 必要性

保留 EvidenceBank，屏蔽 intent。

若性能几乎不变：

```text
global intent contract 没有实际作用
```

## 35.4 Contract 角色独立性

检查：

\[
\cos(g,S_0),\quad \cos(g,Q_0),\quad \cos(S_0,Q_0)
\]

仅作为诊断，不增加 decorrelation loss。

## 35.5 Time disentanglement

固定 observation/intent，改变 diffusion time。

要求：

```text
intent contract 不变
MMDiT modulation 随 time 变化
Stage seed 不随 time 变化
```

## 35.6 Active module completeness

正式新 class 中：

```text
所有 trainable 参数必须进入 forward
所有 intended trainable 参数必须获得梯度
不存在 state-dict-only trainable module
```

---

# 36. 验收标准

## 结构验收

正式 active 主线中必须满足：

```text
无 prior
无 posterior
无 Gaussian sampling
无 KL
无 posterior reconstruction
无 target_physical model input
无 z_to_token
无 CVAE action stem
无 LatentCVAEActionBlock
无 active latent_cvae_* config
无 CVAE class inheritance
```

## 职责验收

```text
Intent Compiler：高层确定性意图合同
EvidenceBank：底层事实
Stage Memory：当前阶段积累
Action State：当前求解状态
Flow noise：多模态随机性
MMDiT：动作向量场
```

## 性能验收

至少使用相同数据、相同 seeds 比较：

1. full RMSE；
2. arm RMSE；
3. gripper field RMSE；
4. first / first4 / first8；
5. tail；
6. decoded action；
7. rollout；
8. event coverage；
9. OOD perturbation；
10. 实际 rollout success；
11. train throughput；
12. peak memory；
13. checkpoint size；
14. 每步 refinement gain。

建议最低门槛：

```text
ID 指标不出现超过统计波动的系统性下降
rollout / OOD 不下降
EvidenceBank 依赖高于 V75
训练计算和 state dict 明显简化
```

不应只根据单一 MSE 决定是否通过。

---

# 37. 风险与回退

## 风险 1：Variational 分支确实提升了条件表示

回退：

```text
保留 B1 deterministic legacy mapping 作为稳定恢复点
```

然后判断收益来自：

- posterior supervision；
- prior MLP 深度；
- latent bottleneck；
- 还是旧 stem。

不要直接恢复整个 CVAE。

## 风险 2：ICC 表示能力不足

升级顺序：

```text
增加 role-specific MLP capacity
→ 保留更多高层 summary
→ 最后才启用 Role Query Compiler
```

## 风险 3：Intent contract 成为新捷径

处理：

- 不读取 full trajectory / rollout；
- 不进入 action seed；
- action-only output；
- 严格 source ownership；
- evidence ablation probe；
- 限制 contract 数量与维度；
- 不引入 target teacher。

## 风险 4：移除 legacy stem 后性能断崖

处理：

- 测量 stem effect；
- 增强 condition-neutral temporal initializer；
- 保留同尺度初始化；
- 逐项对照 stem 的 temporal structure 与 condition bias；
- 不允许新 initializer 重新读取 intent。

## 风险 5：多合同仍然高度相关

先诊断，不立即加额外 loss。

如果长期塌缩，优先通过：

- 不同输入 source mask；
- 不同输出 shape；
- 不同使用位置；
- 独立最后投影；

建立结构性分工，而不是增加 decorrelation loss。

---

# 38. 推荐开发顺序

```text
CR0
残余审计与 probes
    ↓
CR1
Variational 旁路，保持 p_mu 部署函数
    ↓
CR2
意图 / time / stage 接口解耦
    ↓
CR3
ICC shadow path
    ↓
CR4
ICC 接管 Stage / Workspace
    ↓
CR5
ICC 接管 MMDiT modulation
    ↓
CR6
Action State Initializer 接管 legacy stem
    ↓
CR7
清理 output shortcut
    ↓
CR8
建立独立 HierarchicalMMDiTActionDecoder
    ↓
CR9
配置、日志、checkpoint、README 清理
    ↓
重新基于干净主线实现
Exhaustion-Driven Stage-Adaptive Refinement
```

---

# 39. 版本与恢复点建议

不要立即修改已有 V75/V76 标签。

使用独立恢复点：

```text
cvae-r0-audit
cvae-r1-deterministic-legacy
cvae-r2-contract-interface
cvae-r3-icc-shadow
cvae-r4-icc-workspace
cvae-r5-icc-mmdit
cvae-r6-action-initializer
cvae-r7-clean-decoder
cvae-r8-source-clean
```

每个恢复点都保留：

- checkpoint；
- config snapshot；
- metric report；
- parameter diff；
- forward graph；
- source ownership table。

---

# 40. 最终架构定义

最终不再存在一个模糊的 CVAE latent \(z\)。

结构由四类明确状态组成：

```text
g：
稳定、确定性的全局意图控制

L_k：
当前 refine step 的临时底层证据

S_k：
跨 refine step 的阶段记忆

A_k：
动作求解状态
```

动作生成由 flow noise 提供随机性：

\[
x_0\sim\mathcal N(0,I)
\]

MMDiT 负责条件向量场：

\[
A_{k+1}
=
B_{i_k}
\left(
A_k,L_k,S_k,x_t;
g,t
\right)
\]

最终原则：

> **Intent 只决定求解器如何工作，不直接给出动作；Evidence 决定具体事实；Stage Memory 记录当前进度；MMDiT 在 flow noise 上完成多模态动作求解。**

---

# 41. 外部设计依据

本方案没有直接复制某一篇现成机器人 policy，而是组合了几条成熟实践：

1. **Perceiver**：使用非对称 cross-attention 将高维输入蒸馏到紧凑 latent，说明固定数量接口表示是可行的。
2. **Set Transformer / PMA**：使用 learned seed vectors 对集合进行结构化聚合。
3. **BLIP-2 Q-Former**：使用 learnable queries 提取固定数量、与下游任务相关的特征。
4. **DiT / AdaLN conditioning**：条件可以通过 modulation 驱动去噪 backbone，而不必成为动作 token。
5. **Decoupled Action Expert**：近期实验表明，扩散式机器人策略的 task-specific knowledge 可以主要放在 conditioning pathway，action backbone 可以相对 task-agnostic；该工作还报告 modulation-based conditioning 在其解耦设置中优于若干 attention-based conditioning 方式。
6. **Diffusion Policy**：条件动作扩散能够表达多模态动作分布，因此不需要再依赖 CVAE latent 作为第二套部署随机源。

这些工作支持“明确的条件接口 + 生成式动作 backbone”的方向，但本项目仍需通过上述消融确认最适合当前 VLA 数据与 hierarchical workspace 的具体实现。

---

# 42. 最终决策

主方案确定为：

> **使用无随机、无 prior/posterior、无额外 loss 的 Intent Contract Compiler，输出 Global Intent、Stage Seed 和 Evidence Read Anchor 三类显式合同；同时使用 condition-neutral Action State Initializer 接管 legacy CVAE action stem。**

Query-based compiler 只作为容量不足时的升级方案。

这不是删除 CVAE，而是完整取代：

| CVAE 当前职责 | 新结构 |
|---|---|
| 条件压缩 | Policy Condition Organizer |
| 全局意图 | Global Intent Contract |
| Stage 初始化 | Stage Seed Contract |
| Workspace 查询 | Evidence Read Anchor |
| MMDiT 调制 | Intent + Time + Stage modulation |
| 初始 action 加工 | Condition-neutral Action State Initializer |
| 多模态 | Action flow noise |
| target supervision | 仅保留主 policy loss |
| posterior/KL | 不再需要 |
