# V73 设计稿：货架正名第一轮（A 版轻清理）
# —— transition_event 席位换真证据 + scan/lateral 撤架归位 cond

状态: 设计评审中（待批准后实现）
范围: 本轮 = "A 版轻清理"。两项内容变更, 均只动货架、不动 cond 通路:
  (1) transition_event 席位换真事件证据(§3);
  (2) scan/lateral 撤出货架, 全额保留在 primary_cond(§3.5)。
  两项各挂独立 flag, 实验作为一组开启, 回归时可单 flag 隔离归因。
前置: v71/v72 跑完 E3 并选出基座臂
原则依据: 货架纪律二条 —— (1) 世界证据原则(v72 立): 动作写过的内容不得回架;
(2) 名实相符原则(本轮立): 席位内容必须配得上席位语义, 不许拿衍生物顶替。

---

## 1. 背景与动机

短时记忆清点(见会话记录)发现 D2: fixed_zero 模式下

```
transition_memory = [controlled_delta, event_context]          # policy_v39.py:5410
event_context     = _rollout_tokens_to_action_horizon(controlled_delta)   # :5373, 定义 :580
```

`_rollout_tokens_to_action_horizon` 是纯 pooling + 时间对齐(无参数、无新信息)。
即 **transition_event 席位放的是 controlled_delta 的粗粒度复本** —— 5403 行的注释
修掉了 delta/effect 重复, 却漏掉了 delta/event 这条 reshape 通道。

后果:
- 模型在"何时开合"问题上, 货架里没有任何时机语义可读;
- 我们读 wtrans/wtrans_event 份额时以为事件证据被消费, 实为误报, 污染 H4 解释链;
- 夹爪长时域 rmse 是全系统唯一饱和瓶颈(v69 E8: grip_tail 0.174 vs arm_tail 0.080),
  时机证据缺位是头号结构性嫌疑。

## 2. 现状勘查（全部核对过源码）

| 事实 | 位置 |
|---|---|
| 事件标签: `gripper_event_labels(target_raw, current_raw, gripper_index, threshold)` → [B,T] 三类(0=hold, ±1=开/合) | runtime_v36_3.py:105, 242 |
| 最终事件监督: CE on `output["event_logits"]` [B,T,3] + event emphasis 加权夹爪流 | runtime_v36_3.py:440, 340 |
| 主干侧已有时机潜变量: `RecurrentMilestoneConsequenceCell` 输出 `milestone_policy_time_tokens` [B,K,h] | policy_v39.py:1083 |
| 同 cell 输出中性分支(无动作条件): `milestone_neutral_latent_pred` [B,K*grid,h] | policy_v39.py:1084 |
| 逐层入账: `layer_entry["policy_effect_time_tokens"]`, `layer_entry["neutral_latent_pred"]` | policy_v39.py:5259, 5262 |
| 层级事件探针: `event_probe(LN+Linear(h,3))` 读 time_tokens (`layer_causal_event_from_effect=1`) | policy_v39.py:5073, 5308-5310 |
| 对齐工具: `_align_milestone_tokens_to_horizon` / `_rollout_tokens_to_action_horizon` | policy_v39.py:558, 580 |
| 解码器摄入: `_memory_tokens(transition_source, traj, transition_proj)` 每个 list 元素池化为 1 token; 席位切片 [0:1]=delta, [1:2]=event | policy_v39.py:3525, 3598-3609 |
| 货架 detach 约定: `latent_cvae_transition_detach=1`, transition/rollout 证据默认断梯度入架 | policy_v39.py:173, 3521-3524, 3634-3637 |
| 货架支持任意 [B,N,h] 源, −log(count) 组先验自动均衡 | policy_v39.py:2176 |

关键机制注意: consequence cell 的输入含 `action_physical=consequence_physical`
(默认 = noisy_physical, 即 x_t; policy_v39.py:5176, 5227)。因此 **policy/intervention
分支的一切输出都是 x_t 条件化的**; neutral 分支是无动作条件的场景驱动预测。

## 3. 设计

### 3.1 方案 B（主案）: 中性分支事件证据头

**内容主张**: 货架上的事件证据应回答"场景说抓取窗口在哪"(世界证据),
而不是"当前候选动作打算何时抓"(那是解码器自己的工作, 它有 x_t)。
grab-pen 的事件时机本质上是场景驱动的(到达笔的位置 → 闭合窗口)。

**结构**(全部在主干侧, 生产者在消费者上游):

```
neutral_latents = layer_contracts[-1]["neutral_latent_pred"]        # [B, K*grid, h], x_t-free
event_anchor    = _rollout_tokens_to_action_horizon(neutral_latents)  # [B, horizon, h] 复用 :580
event_evidence  = EventEvidenceHead(event_anchor)                     # 新模块, 见下
```

`EventEvidenceHead`(新, 挂在 V39PolicySystem/planner 层级):
```
lift:   LayerNorm(h) → Linear(h, h) → SiLU → Linear(h, h)   # 类型化提炼
logits: Linear(h, 3)                                         # 监督探针(共享输出)
```
- `event_evidence_tokens = lift(event_anchor)`  [B, horizon, h]
- `event_evidence_logits = logits(event_evidence_tokens)` [B, horizon, 3]

**监督**: CE(event_evidence_logits, gripper_event_labels), 类权重沿用最终事件头
的处理(hold 类占绝对多数, 需类平衡或沿用 event emphasis 的 mask 加权),
权重 `event_evidence_loss_weight`(新 trainer 配置, 默认 0.1)。
监督是这一席"名实相符"的强制机制: 席位内容被训练成必须携带真时机信息。

**上架**(替换伪席位):
- `transition_memory = [controlled_delta]`(event_context 撤下, delta 席保留);
- `evidence_sources["transition_event"] = event_evidence_tokens.detach()`
  ([B,horizon,h] 全时序分辨率直接上架, 不经 `_memory_tokens` 池化 ——
  时机信息的价值恰在逐步结构; trajectory 已有全分辨率上架先例)。
  detach 遵循 `latent_cvae_transition_detach=1` 约定: 证据头的梯度只来自
  自己的 CE, 不来自动作损失 —— 动作不能重塑自己的证据(防套利),
  证据质量由监督保证(名实相符)。

### 3.2 方案 A（备选, 更廉价）: 复用 policy_effect_time_tokens

`event_evidence_tokens = align(layer_contracts[-1]["policy_effect_time_tokens"])`,
仅加 lift, 不加新监督(已有 event_probe CE)。
代价: 内容是 x_t 条件化的 —— 训练期 x_t 在低 t 含 oracle 时机, 存在
teacher-leak 进证据的风险(与 mdna 低 t 问题同构), 且部署期证据质量
随自身去噪状态波动。**不推荐为主案**, 保留为对照变量(见 §6)。

### 3.3 flag 与兼容

- `latent_cvae_event_evidence: int = 0`(config + validate + CLI
  `--latent-cvae-event-evidence`)。0 = 现状(伪席位), 1 = 方案 B。
- flag=0 时 EventEvidenceHead 不实例化, 参数量与 v72 严格一致, checkpoint 兼容。
- 旧 event_context 路径整体保留在 else 分支, git tag 照常。

### 3.5 A 版轻清理: scan/lateral 撤架归位 cond

**语义主张**(通路正名): 货架 = 世界证据, cond/z = 任务语义。
scan/lateral 是中高层语义摘要, 归属 cond 通路 —— 它们在 3500-3510 行本就
全额构成 primary_cond(每步喂给解码器), 货架上的两席是同一内容的二次入账。
撤架信息零损失; 将来接入更高层语义输入(VLM/语言)时落点在 cond 侧,
货架结构不动。

**改动**: `latent_cvae_workspace_semantic_dedup: int = 0`(flag)。
=1 时 evidence_sources 组装(policy_v39.py:3511, 3537-3538)跳过
`lateral` 与 `scan` 两个键; primary_cond 组装路径一行不动。

**市场影响与判读基线**: 九席 → 七席(v73 两项合计: −lateral −scan,
transition_event 换真货), −log(count) 组先验自动重归一。
与 v72 的份额对比需按新席位数重新基线, wgeff 预期上升
(v72 末约 5.3, 失去两个低份额源后有效源数的相对值应更接近满额)。
市场对撤架的预期反应: 原 0.036+0.065 ≈ 0.10 的注意力质量重新分配 ——
分配去向本身是信息(流向 transition_event 为佳, 流向 routed_layer 中性)。

**风险**: 接近零 —— 市场已把两席定价到底部, 内容在 cond 里全额存活。
若 val 意外劣化, 说明这两席存在未被理解的非语义功能, 单 flag 关回即可,
且这个"意外"本身值得专项调查。

## 4. 反回声 / 反套利 / 反泄漏审计（设计自检）

1. **回声**: 生产者(主干 consequence cell 中性分支 + 新头)严格位于消费者
   (workspace/MMDiT)上游, 同趟前向无回流; 上架 detach。通过。
2. **套利**: 席位内容由独立 CE 锚定, 动作损失无法通过重塑证据换取 loss;
   若模型试图忽略该席, wtrans_event 份额直接可读。通过。
3. **teacher leak**: neutral 分支无 x_t 输入; 标签来自数据集夹爪轨迹,
   与 val/deploy 同分布。CE 监督的是"场景→时机"映射, 部署期成立。通过。
4. **名实相符**: 席位内容 = 受监督的场景驱动事件时间表。通过。

## 5. 仪表与判读

已有仪表直接覆盖, 无需新增:
- `wtrans_event`(workspace_transition_event_attention): 席位换真货后市场是否加价
  —— 若仍趴在先验附近, 说明证据无增量或市场失灵, 两种情况都值得知道;
- `gfnehr`: H4 判决表 —— 显式时机先验上架后, 事件步 null(延迟视图时机分歧)
  应收缩; 若 gfnehr 回落向 1 以下且夹爪 val 改善, H4 的"时机不确定性签名"
  获得干预性证据(不止相关性);
- `mdwaT` 低 t 桶: 世界证据在低 t 区间的份额应上升(4:1 的 x_t:workspace 应收窄);
- 判决主指标: **gripper_tail_rmse / gripper_full_rmse**(唯一饱和通道);
- 新增仅一个: `event_evidence_acc`(证据头 CE 的 argmax 对标签准确率, 训练侧,
  确认证据头本身学到了东西 —— 若它自己都预测不了时机, 席位判读无从谈起)。

## 6. A/B 协议

- 基座: v71/v72 E3 判决的胜者臂(预期 v72)。
- v73 = 基座 + `--latent-cvae-event-evidence 1`
  + `--latent-cvae-workspace-semantic-dedup 1`(一组实验、两个 flag;
  回归时以单 flag 复跑隔离归因)。
- 脚本 `scripts/current_v73_event_evidence.sh`, 链式 wrapper, 头注释含判读规则。
- 跑到 E3 做首次结算; E1 只看 event_evidence_acc 是否爬升与 canary。
- 可选第三臂(资源允许时): 方案 A 作源模式对照, 分离"有时机证据"与
  "时机证据是否可被 x_t 污染"两个因素。

预期读数(按假说方向):
- event_evidence_acc: 快速升至高位(hold 类基线很高, 看事件类召回);
- wtrans_event: 从先验(~0.02, 1/9 组先验内的份额)显著上抬;
- gfnehr: 晚 epoch 的 ≥1 峰值回落; gripper_fm_event(gfme)下降;
- val: gripper_tail/full 改善, arm 不动 —— 若 arm 也动, 说明有意外串扰, 停下来查。

## 7. 实现清单

1. `policy_v39.py`:
   - config: `latent_cvae_event_evidence`、`latent_cvae_workspace_semantic_dedup`
     + validate 列表;
   - evidence_sources 组装: dedup flag 跳过 lateral/scan 两键(3511/3537-3538);
   - `EventEvidenceHead` 模块(lift + logits, ~2.4M 参数 @ h=768);
   - 系统 forward: flag 分支 —— 中性潜变量对齐 → 头 → 
     `transition_memory` 改为 `[controlled_delta]` + 独立 `event_evidence` 传参;
   - 解码器 forward/条件组装: 新参数 `event_evidence_tokens`, 
     `evidence_sources["transition_event"] = tokens.detach()`(flag=0 走旧切片);
   - 输出: `event_evidence_logits` 入 output dict 供 loss 用。
2. `policy_runtime_v36_3.py`: `event_evidence` CE loss(类平衡)、
   `event_evidence_acc` 指标、trainer 配置 `event_evidence_loss_weight`。
3. `policy_runtime_v39.py`: loss 键转发 + console `evac=`/`wtev=` 
   (wtrans_event 已有, 确认显示)。
4. `train_v40_policy.py`: CLI 两个参数(flag + loss weight)。
5. `scripts/current_v73_event_evidence.sh`。
6. 验证: py_compile ×4 + bash -n + 一致性 grep
   (flag 两态的 transition_memory 组成、席位 ranges、console 键)。

## 8. 风险与回退

- **R1 中性潜变量质量不足**(神经元没学到场景→时机): event_evidence_acc 
  不上升即为证据, 此时换方案 A 或将头输入改为 canvas trajectory slice。
  判读不受污染 —— acc 是独立读数。
- **R2 类失衡塌缩**(全预测 hold): 类权重/事件加权处理, acc 按事件类单独报。
- **R3 市场不买账**(wtrans_event 不动): 席位真货但决策增量为零 ——
  本身是对 H4 的信息; 不构成回退理由, 构成解释。
- **回退**: flag 置 0 即完全恢复 v72 行为, 无 checkpoint 迁移成本。
