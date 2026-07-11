# V76：Exhaustion-Driven Stage-Adaptive Refinement

## 1. 目标

V76 的目标不是训练一个额外的 gate，而是让 refinement 过程根据自身的运行状态决定：

- **stay**：继续使用当前 block；
- **advance**：当前 block 已经枯竭，切换到下一个 block；
- **exit**：动作与证据均已稳定，结束 refinement。

核心思想：

> 将 block 数量与最大 refinement step 数解耦。  
> block 表示不同类型的更新算子，step 表示最大计算预算。  
> 阶段停留、阶段切换和最终退出均由主过程自身的“枯竭状态”决定。

该设计不引入 learned routing branch，也不增加 halting loss。

---

## 2. 结构定义

设共有三个不同职责的 refinement block：

\[
B_1,\quad B_2,\quad B_3
\]

最大 refinement 预算为：

\[
K_{\max}=6
\]

每个 block 可以被重复调用若干次，且阶段只允许单调前进：

\[
B_1 \rightarrow B_2 \rightarrow B_3
\]

允许的状态转移为：

- 在 \(B_1\)：stay / advance / exit；
- 在 \(B_2\)：stay / advance / exit；
- 在 \(B_3\)：stay / exit；
- 不允许阶段回退。

因此实际执行路径可以是：

```text
B1 → B2 → exit
B1 → B2 → B3 → exit
B1 → B1 → B2 → B3 → exit
B1 → B2 → B2 → B3 → B3 → exit
B1 → B1 → B2 → B2 → B3 → B3
```

---

## 3. 四个核心公式

### 3.1 随机阶段停留训练

\[
\boxed{
\hat a
=
H\!\left(
B_3^{\,d_3}
\circ
B_2^{\,d_2}
\circ
B_1^{\,d_1}
(A_0)
\right),
\qquad
(d_1,d_2,d_3)\sim\mathcal D,
\quad
d_i\ge1,
\quad
\sum_i d_i\le K_{\max}
}
\]

其中：

- \(B_1,B_2,B_3\)：三个不同职责的 MMDiT refinement block；
- \(d_i\)：当前训练样本在第 \(i\) 个 block 上的停留次数；
- \(H\)：现有 physical action head；
- \(K_{\max}=6\)。

训练路径不再固定为：

\[
(d_1,d_2,d_3)=(1,1,4)
\]

而是从一个单调 schedule 分布中采样，例如：

```text
(1,1,1)
(1,1,2)
(1,2,1)
(2,1,1)
(1,2,2)
(2,1,2)
(2,2,2)
(1,1,4)
```

训练仍然只使用现有 policy loss：

\[
\mathcal L=\mathcal L_{\text{policy}}
\]

不增加 halting loss 或 router supervision。

---

### 3.2 物理动作响应

当前 block 对物理动作仍能造成多大有效改变：

\[
\boxed{
u_k
=
\frac{
\left\|
\hat a_{k+1}-\hat a_k
\right\|_{\mathcal M}
}{
\left\|
\hat a_k
\right\|_{\mathcal M}
+\varepsilon
}
}
\]

其中：

\[
\hat a_k=H(A_k,c)
\]

\(\|\cdot\|_{\mathcal M}\) 不使用普通 18 维欧氏 MSE，而是复用当前 policy 的物理语义度量：

\[
\frac{
6\text{ 个 arm 维度}
+
1\text{ 个 gripper field}
}{7}
\]

并保持现有 manifold、Parseval field 和 null-space 语义。

解释：

```text
u_k 高：当前 block 仍然能够有效改变物理动作
u_k 低：当前 block 的动作更新已经接近枯竭
```

---

### 3.3 阶段证据压力

Stage memory 是否仍在吸收新的证据信息：

\[
\boxed{
p_k
=
\frac{
\left\|
S_{k+1}-S_k
\right\|_F
}{
\left\|
S_k
\right\|_F
+\varepsilon
}
}
\]

解释：

```text
p_k 高：阶段证据仍在发生明显重组，当前问题尚未稳定
p_k 低：阶段证据基本稳定
```

当前 `HierarchicalEvidenceWorkspace` 已经存在类似：

```text
hierarchical_stage_update_norm
```

但当前实现通常只保留 batch mean。V76 需要保留 per-sample ratio。

---

### 3.4 唯一路由规则

\[
\boxed{
r_k=
\begin{cases}
\mathrm{stay},
&u_k>\tau_u
\\[2mm]
\mathrm{advance},
&u_k\le\tau_u,\ p_k>\tau_p,\ i_k<M
\\[2mm]
\mathrm{exit},
&u_k\le\tau_u,\ p_k\le\tau_p
\\[2mm]
\mathrm{exhausted},
&u_k\le\tau_u,\ p_k>\tau_p,\ i_k=M
\end{cases}
}
\]

其中：

- \(i_k\)：当前 block 阶段；
- \(M\)：最后一个 block 的索引；
- \(\tau_u\)：动作响应阈值；
- \(\tau_p\)：证据压力阈值。

对应语义：

| 动作响应 \(u_k\) | 证据压力 \(p_k\) | 当前状态 | 决策 |
|---|---:|---|---|
| 高 | 任意 | 当前 block 仍有效 | stay |
| 低 | 高 | 当前 block 已枯竭，但证据尚未稳定 | advance |
| 低 | 低 | 动作与证据均已稳定 | exit |
| 低 | 高，且已在最后一个 block | 所有算子已耗尽，但问题仍未稳定 | exhausted |

最后一种情况记录：

```text
exhausted_unresolved = True
```

第一版只作为诊断指标，不直接改变控制行为。

---

## 4. 核心执行逻辑

```text
EvidenceBank
    ↓
Workspace 生成 L_k 与 S_k
    ↓
当前 block B_i 更新 A_k
    ↓
Physical Action Head 解码 â_k
    ↓
计算动作响应 u_k
    ↓
计算证据压力 p_k
    ↓
Exhaustion Rule
    ├─ stay
    ├─ advance
    └─ exit
```

该过程不存在：

- learned route logits；
- task-conditioned halt classifier；
- action-to-workspace-manager feedback；
- auxiliary halting objective；
- 独立 routing branch。

---

## 5. 实施阶段

## 5.1 V76A：Exhaustion Probe

### 目标

不改变现有 forward 行为，只验证内部指标是否真的能够预测 refinement 收益。

现有执行路径保持：

```text
B1 → B2 → B3 → B3 → B3 → B3
```

每一步额外记录 per-sample：

```text
step_action_response
step_stage_pressure
step_physical_prediction
step_block_id
```

对应：

\[
u_k
\]

\[
p_k
\]

以及每一步的物理动作预测：

\[
\hat a_k
\]

训练仍然只使用最终 policy loss。

验证阶段额外计算：

\[
\Delta E_k=E_k-E_{k+1}
\]

其中 \(E_k\) 是第 \(k\) 步动作预测相对于 target 的真实误差。

该量仅用于诊断，不参与反向传播。

### 需要验证的问题

1. \(u_k\) 高时，下一步是否通常仍有真实收益；
2. \(u_k\) 低且 \(p_k\) 低时，后续 refinement 是否基本无收益；
3. \(u_k\) 低但 \(p_k\) 高时，切换 block 是否优于重复当前 block；
4. \(u_k,p_k\) 是否只是绝对 step index 的代理；
5. 阈值在 train / validation 之间是否稳定。

### 通过条件

至少满足：

- 动作响应与真实边际改善存在明确统计关系；
- 证据压力能够区分“已完成”和“当前 block 卡住”；
- 不同任务、样本和 diffusion time 下存在非固定路径趋势；
- 指标分布不会严重依赖训练集 task identity。

---

## 5.2 V76B：Randomized Dwell Training

### 目标

让三个 block 真正学会在不同停留时间下工作，使 block identity 与绝对 step 解耦。

当前固定索引：

```python
mmdit_block = self.mmdit_blocks[
    min(step, len(self.mmdit_blocks) - 1)
]
```

改为显式 schedule：

```python
schedule = sample_monotonic_schedule(
    block_count=3,
    max_steps=6,
)

for block_id in schedule:
    action = self.mmdit_blocks[block_id](...)
```

### 初始 schedule 分布

建议第一版按 batch 共享一个 schedule，避免动态 shape、DDP 和 compile 复杂度。

```text
50%：当前路径附近
     (1,1,4)
     (1,2,3)
     (2,1,3)

30%：均衡路径
     (2,2,2)
     (1,2,2)
     (2,1,2)

20%：短路径
     (1,1,1)
     (1,1,2)
     (1,2,1)
```

该阶段仍不增加任何新损失：

\[
\mathcal L=\mathcal L_{\text{policy}}
\]

### 预期结果

模型应学会：

- \(B_1\) 可以重复；
- \(B_2\) 可以重复；
- \(B_3\) 可以重复；
- block identity 不再等于绝对 step；
- 不同停留时间下最终动作仍可正确解码。

---

## 5.3 V76C：Shadow Adaptive Routing

### 目标

在不改变真实执行路径的情况下，离线验证 exhaustion rule。

真实执行仍使用固定或随机完整路径，但同时运行 shadow route：

```text
真实执行：
B1 → B2 → B3 → B3 → B3 → B3

影子判断：
B1 → B1 → B2 → B3 → exit
```

记录：

```text
shadow_block_path
shadow_exit_step
shadow_unresolved
shadow_average_steps
```

需要比较：

- shadow route 的理论平均步数；
- shadow exit 对应的动作误差；
- 固定 3 / 4 / 5 / 6 步的动作误差；
- 不同任务和 diffusion time 的退出分布；
- 是否退化成固定路径；
- 是否经常出现 unresolved 状态。

该阶段不影响模型输出，也不改变训练。

---

## 5.4 V76D：真实自适应执行

### 状态

```python
block_id = 0
active = True
consecutive_exhaustion = 0
```

### 逻辑示例

```python
candidate_action = blocks[block_id](...)

u = physical_action_response(
    candidate_action,
    action,
)

p = stage_pressure(
    next_stage,
    stage,
)

action = candidate_action
stage = next_stage

if u > tau_u:
    consecutive_exhaustion = 0

elif p > tau_p and block_id < last_block:
    block_id += 1
    consecutive_exhaustion = 0

else:
    consecutive_exhaustion += 1

if consecutive_exhaustion >= 2:
    break
```

使用连续两次枯竭确认：

```text
two-step exhaustion confirmation
```

避免单步波动导致过早退出。

阶段转移严格单调：

```text
B1 → B2 → B3
```

禁止：

```text
B3 → B2
B2 → B1
```

---

## 6. 当前代码需要修改的位置

## 6.1 Workspace 返回 per-sample stage pressure

当前类似实现：

```python
"hierarchical_stage_update_norm": (
    next_stage_content.detach().float()
    - stage_content.detach().float()
).norm(dim=-1).mean()
```

应同时返回 per-sample tensor：

```python
stage_delta = (
    next_stage_content.float()
    - stage_content.float()
)

stage_pressure_rows = (
    stage_delta.square()
    .mean(dim=(1, 2))
    .sqrt()
    /
    stage_content.float()
    .square()
    .mean(dim=(1, 2))
    .sqrt()
    .clamp_min(1e-6)
)
```

日志仍可记录：

```python
stage_pressure_rows.mean()
```

但 adaptive route 使用：

```python
stage_pressure_rows
```

---

## 6.2 每一步调用共享 physical action head

当前通常只在 refine loop 完成后调用：

```python
out = self._emit_action(...)
```

V76A 中每一步增加 detached probe：

```python
step_action = self._emit_action(
    self.mmdit_action_norm(action),
    primary_cond,
)["pred_velocity"]
```

注意：

- 只对临时输出 tensor 使用 `mmdit_action_norm`；
- 不把 normalized action 写回 recurrent state；
- 所有 step 共享同一个 physical action head；
- probe 不增加 loss。

---

## 6.3 去除 block 与绝对 step 的硬绑定

当前：

```python
block_id = min(step, depth - 1)
```

改为：

```python
block_id = schedule[step]
```

真实 adaptive inference 中改为：

```python
block_id = current_stage
```

---

## 6.4 重构 workspace step embedding

当前如果使用：

```python
self.step_embedding[:, step_index]
```

容易重新建立：

```text
absolute step → fixed evidence reading strategy
```

建议改为：

```text
stage embedding
+
remaining-budget embedding
```

例如：

```python
step_state = (
    self.stage_embedding[:, block_id]
    + self.budget_projection(
        remaining_steps / max_steps
    )
)
```

这样：

- `block_id` 表示当前算子的职责；
- remaining budget 只表示剩余计算预算；
- 不再暗示“第三步必然属于 B3”。

---

## 7. 阈值标定

\(\tau_u\) 和 \(\tau_p\) 不参与训练，不接收梯度。

建议先按 diffusion time 分区间标定：

```text
t ∈ [0.00, 0.33)
t ∈ [0.33, 0.66)
t ∈ [0.66, 1.00]
```

因为不同噪声阶段的 action response 与 stage pressure 量级可能明显不同。

标定目标不是单纯追求最低平均 step，而是满足：

```text
相对固定 6 步：
性能下降 ≤ 1%

同时：
平均 refinement steps 明显下降
unresolved 比例可解释
不同任务产生不同路径
路径不退化为 step lookup
```

---

## 8. 防捷径约束

V76 必须继续保持现有 firewall：

```text
Action ──╳──> Workspace Manager
Action ──╳──> EvidenceBank K/V
```

允许：

```text
Action output → physical action response probe
Stage memory → stage pressure probe
```

但 probe：

- 不参与 evidence value 构造；
- 不改变 workspace routing；
- 不接收独立 routing loss；
- 第一阶段全部 `detach()`；
- 只作为主过程运行状态的观测量。

---

## 9. 主要日志指标

建议新增：

```text
exh_u_mean
exh_u_p25
exh_u_p50
exh_u_p75

exh_p_mean
exh_p_p25
exh_p_p50
exh_p_p75

exh_shadow_exit_step
exh_shadow_block_id
exh_shadow_unresolved_rate

exh_gain_corr
exh_step_histogram
exh_path_histogram
```

其中：

- `exh_gain_corr`：\(u_k\) 与真实下一步改善 \(\Delta E_k\) 的相关性；
- `exh_path_histogram`：不同 block schedule 的实际分布；
- `exh_shadow_unresolved_rate`：最后一个 block 枯竭但证据仍未稳定的比例。

---

## 10. 必须完成的消融

### 10.1 固定深度基线

```text
B1 → B2 → B3 ×4
```

### 10.2 单一共享 block

```text
B ×6
```

用于验证多个 block 是否真的承担不同阶段职责。

### 10.3 随机 dwell，无 adaptive route

验证随机停留训练本身是否改善 anytime robustness。

### 10.4 只看动作响应

\[
u_k
\]

验证单一 convergence criterion 是否会混淆“完成”和“卡住”。

### 10.5 动作响应 + 证据压力

\[
u_k,\quad p_k
\]

验证 stage pressure 是否能正确触发 advance。

### 10.6 Learned gate 对照

可以保留一个小型 learned halt head 作为消融基线，但不作为主方案。

用于证明：

- learned gate 是否更容易退化为 task / step shortcut；
- exhaustion-driven route 是否具有更好的跨任务稳定性；
- 两者在平均计算量相同时的性能差异。

---

## 11. 风险与回退方案

### 风险 1：动作响应低但真实误差仍高

解释：

```text
当前 block 卡住
```

处理：

- 检查 stage pressure；
- 若 \(p_k\) 高，则 advance；
- 若已在最后一个 block，则标记 unresolved。

### 风险 2：Stage memory 持续振荡

处理：

- 使用两步确认；
- 对 \(p_k\) 使用短窗口 EMA；
- 检查 stage update 是否被 noisy evidence 驱动。

### 风险 3：指标退化为 step proxy

处理：

- 随机 dwell training；
- 使用 stage embedding 替代 absolute step embedding；
- 按 task / time / schedule 分组检查指标分布。

### 风险 4：短路径训练不足

处理：

- 第一阶段保持长路径占比更高；
- 逐步增加短路径概率；
- 不立即启用真实 early exit。

### 风险 5：真实动态执行影响 DDP 或 compile

第一版只做：

```text
batch-level break
```

或逻辑冻结，不做 active batch compaction。

待效果确认后再实现 per-sample active batch 压缩。

---

## 12. 推荐开发顺序

```text
V76A
被动 probe
    ↓
验证 u_k / p_k 是否有效
    ↓
V76B
随机阶段停留训练
    ↓
解除 block 与绝对 step 的绑定
    ↓
V76C
shadow adaptive routing
    ↓
离线标定 tau_u / tau_p
    ↓
V76D
真实自适应推理
    ↓
最后再考虑 active batch compaction
```

在 V76A 和 V76C 未通过之前，不启用真实动态执行。

---

## 13. 论文命名

推荐名称：

> **Exhaustion-Driven Stage-Adaptive Refinement**

完整表述：

> **Monotonic Exhaustion-Driven Stage-Adaptive Action Refinement**

中文：

> **枯竭驱动的单调阶段自适应动作精炼**

---

## 14. 论文核心描述

> The policy decouples operator depth from iteration budget and determines stage residence, stage transition, and termination from the joint exhaustion of physical action response and recurrent evidence update, without a learned routing branch or auxiliary halting objective.

中文表述：

> 该策略将更新算子的参数深度与迭代计算预算解耦，并根据物理动作响应与循环证据更新的联合枯竭状态，统一决定阶段停留、阶段切换和最终终止，而不引入学习式路由分支或辅助停机目标。

---

## 15. 核心总结

```text
动作仍有明显响应
→ stay

动作响应枯竭，但阶段证据仍未稳定
→ advance

动作响应与阶段证据同时枯竭
→ exit
```

最终结构为：

```text
Block repertoire:
B1, B2, B3

Maximum budget:
6 refinement steps

Adaptive mechanism:
physical-action-response exhaustion
+
stage-evidence-pressure exhaustion

Available transitions:
stay / advance / exit
```

V76 的核心不是增加一个负责控制主网络的系统，而是：

> 让现有 refinement 过程读取自身是否仍在有效工作，并在当前算子能力耗尽时自然进入下一阶段，在整体计算耗尽时自然终止。
