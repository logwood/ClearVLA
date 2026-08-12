# ClearVLA 主线恢复问题账本

更新：2026-08-12

历史审查对象：独立 mainline schema 20 与 V120 `long`

行为基线：V120 commit `0b92d359a2889a0a1b1eba256007c00ccbc54f3c`

本文件保留 schema 20/21 退化的历史证据、根因和修改权限。Schema 21
虽然恢复了若干 producer/consumer，却把 V120 动态 P1 policy block 放进
了静态 factual query，并给 protected detail 增加了可相消 value 与 learned
null；batch-600 的 factual collapse 已在 schema 22 修正。当前未解决项只看
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md)，不要从本历史
账本重新拼装当前图。

## 1. 审查规则

每一项必须分开写清四层内容：

1. **实现事实**：源码实际计算了什么，不能由类名、注释或契约推断。
2. **实验观察**：日志/探针真正测到了什么，并注明覆盖范围。
3. **因果判断**：只有实现事实和实验观察能相互解释时才升级为结构结论。
4. **修改权限**：分为直接恢复、可直接修复、需先展示三类。

### 1.1 对 V120 的处理原则

| 类别 | 判据 | 处理 |
| --- | --- | --- |
| V120 活跃且有行为证据 | 正式配置启用、数据流真实经过、V120 长跑/梯度支持其有效 | 默认原样或行为等价恢复；不能以“更干净”为理由改输入分布 |
| V120 或适配层的明确缺陷 | future leak、时间帧冒充、单位/方向错误、被声明为 neutral 却由 bias 产生非零、不可成立的轴语义 | 可以直接修；账本必须写旧/新公式和受影响消费者 |
| 合理但证据不足的改进 | 会改变输入统计、梯度几何、attention 竞争、neutral counterfactual 或 source bank | 实现前先向用户展示旧/新路径、预期收益、风险和回退条件 |

任何参数量相近、模块名相同、权重成功加载都不能证明行为等价。所谓“性能优化”默认不得改变：张量含义、动态/静态边界、source 数量与顺序、归一化、残差位置、梯度路径和损失对象。

## 2. 受控证据

### 2.1 完整长跑

三条历史正式日志使用同一任务、batch size 8、24-step action horizon、2846 step/epoch、8 epoch：

| 运行 | E1 physical RMSE | 最好 | E8 | 已确认结论 |
| --- | ---: | ---: | ---: | --- |
| V120 | 0.09762 | 0.07931（E7） | 0.08145 | 当前行为恢复基线 |
| V122 | 0.09760 | 0.08914（E6） | 0.09109 | 近端继续拟合，远端停滞；gripper 越训越保守 |
| schema-19 独立 mainline | 0.10933 | 0.09107（E7） | 0.09127 | 每轮都差于 V120；G/W 对象公共化 |

V120 的结构信号不是完美，但确实存在：G content pair cosine 约 `0.340 -> 0.508`；S interval variation 约 `0.068 -> 0.133`，temporal variation 约 `0.024 -> 0.078`；W2 object pair cosine 约 `0.508 -> 0.445`。因此恢复目标不是只复制 aggregate RMSE，而是恢复这些仍被 action 消费的差异。

### 2.2 schema-20 完整第一轮训练与 validation 崩溃

更新后的 `schema20_recovery_b8_20260811_231427.log` 包含 142 个窗口行，
覆盖 epoch 1 batch 20–2840。训练数据已经走完，但没有任何 epoch/validation
记录：validation 在第五个 batch 进入非诊断路径时抛错。`eval_diagnostic_batches=4`
使前四批带有 execution candidate tensors；第五批关闭可选诊断后，
`execution_value_terms()` 仍无条件读取这四个 loss 必需张量，因而报：

```text
missing evidence_mmd_it_execution_candidate_value_field,
        evidence_mmd_it_dwell_candidate_pred_velocity,
        evidence_mmd_it_execution_candidate_value_mask,
        evidence_mmd_it_execution_baseline_pred_velocity
```

这是 loss-required state 被错误绑定到 logging diagnostic budget 的 runtime
缺陷，不是 OOM、NaN 或训练爆炸。因为没有完成任何 validation，本日志仍不能
给出泛化结论。

第一轮训练本身给出以下决策性证据：

- action flow 从 `1.494` 降到 `0.184`，最低 `0.157`（batch 2760）；native
  tail 为 `0.187`。因此 schema-20 最终可以拟合训练流场，但这不能抵消早期
  相对 V120 的明显分叉，也不能证明 validation 恢复。
- P2 在 batch 160 的 effect RMS 达到 `0.282`，到 batch 420 semantic/
  geometry null mass 同时超过 `0.99`；之后 effect 最低到 `3.1e-9`，batch
  2840 consequence gradient 只有 `9.0e-13`。这是一次清晰的吸收态转变，
  不是“W 稍弱”。
- P3 precision base 从 batch 200 的 `6.77e-3` 降到 tail `4.67e-6`，而
  temporal base 从 `0.331` 增至约 `0.640`。P2/consequence 消失后，动作图
  主要通过 temporal/history 型旁路继续下降。
- controlled transition value 最高 `1.466`（batch 1760），postclip norm
  最高 `0.9994`；tail median 为 `0.853`，约占总 clip 平方预算 73%。global
  preclip tail median `9.39`，最高 `25.89`。schema-20 的静态 transition
  确实重写了优化几何。
- G content pair cosine 从 `0.996` 降至 `0.404`，W object pair cosine 从
  `1.000` 降至 `0.453`：对象槽在第一轮内形成了差异，不能再把失败概括为
  “G/W 全部同质”。真正未形成的是时间区分：W adjacent cosine tail
  `0.982`，prediction interval variation `0.0112`，仍只有 teacher `0.0232`
  的约一半。
- P1 query chart variation tail 仅 `5.34e-4`；在 detail RMS `0.187` 时，
  动作查询仍几乎选择相同空间内容。这与 schema-20 在 query 前收缩 N=49
  fine candidates 的源码事实一致。
- gripper event flow tail `0.621`，约为 hold flow `0.191` 的 3.26 倍；event
  F1 只有 `0.364`。aggregate action flow 的下降掩盖了事件动作仍明显更难。
- visual flow warp `0.153 -> 0.097`、cycle `1.077 -> 0.119`，confidence
  `0.016 -> 0.160`；同时 grid-cell magnitude `0.535 -> 0.087`。它数值上
  变稳定，但没有 moving-region gain 或 action intervention，不能据此声称
  learned flow 已提供动作效益。
- execution value 并非完全失效：tail correlation `0.466`、pairwise accuracy
  `0.754`；但 decision accuracy `0.425`、common-mode ratio `0.282`，说明
  候选排序只学到部分信息。

运行速度中位数 `1.315 s/batch`，进程峰值估计 `4.289 GiB`。这不是纯优化
收益：下面 P0-1 证明一个原本每个 ODE step 依赖 noisy action 的大分支被
提前缓存；P1 也没有执行完整的 query-specific 49-candidate read。

## 3. 已确认的 P0 问题

### P0-1 `DYNAMIC-TRANSITION-WAS-MADE-STATIC`

**实现事实**

- 当前 `policy.py` 在 `encode_online()` 中用 history proposal 构建 transition，并把它放进 `OnlinePolicyCache`；五个 ODE step 只重算 P2/P3/bottom。
- 当前 `transition.py` 计算 `coeff(history_proposal)-coeff(zero_proposal)`。
- V120 在每个 policy forward/ODE step 内以当时的 noisy trajectory/action tokens 计算 `ControlledResidualLatentDynamics`；neutral 使用 learned neutral queries/bias。

这不是机械提取，而是三项行为变化：

```text
V120: per-step noisy action -> dynamic transition -> bottom
当前: observed history proposal -> static transition -> cache -> every ODE step

V120 neutral: learned neutral state/context
当前 neutral: same network with zero history proposal
```

**实验观察**

- transition 成为最大梯度所有者，batch 340 消耗约 79% postclip 平方预算。
- global preclip 从早期约 4 持续升到 9.67；action loss 同期从 V120 曲线分叉。
- 当前显著加速与这个大分支退出五步循环相符。

**判断**

这是已确认的输入语义、动态边界和梯度几何变化，也是当前最强的退化解释。保留 512 行不等于恢复 V120 transition。

**决定**

先恢复 V120 的 per-ODE noisy-action transition 和动态调用位置。`learned neutral` 是否应改成 `zero action` 不是已证明改进；它会改变 counterfactual，必须单独展示后再决定，不能与动态恢复捆绑。

### P0-2 `BOTTOM-EVIDENCE-BANK-DRIFT`

**实现事实**

当前虽然实例化了提取出的 V120 `EvidenceLatentMMDiTActionDecoder`，但其入口已被重新装配：

- generic rollout 保留 selector，却把 rollout value 强制清零；
- `rollout_tokens=transition.selector`，另以 `transition.value` 写入 named transition source；
- P3 role bank 从 V120 的五类事实/precision/effect/temporal/state-change 语义变成三类 `precision/temporal/state_change` 加一个 protected base；
- `layer_contracts` 是新合成的两组 token，而不是 V120 顶层真实执行边界；
- bottom 只收到 current state 和 last executed action 的 generic intent，其他路径由新 S/P/transition 代替。

其中 generic trajectory value 为零、generic history 去重与 V120 object-mainline 基本一致；问题不应笼统写成“所有清零都错”。真正未验证的是 rollout value、source bank 数量/顺序、layer contract 和 transition 角色同时发生改变。

**实验观察**

active bottom 参数和 codec 仍在，但 action 曲线远差于 V120，且 transition 梯度压倒 bottom/P3。说明“V120 decoder 类存在”不足以保证 decoder 看到 V120 分布。

**决定**

以 V120 的真实 decoder 调用逐参数、逐 source 对齐，先恢复 source 数量、顺序、value 统计和 layer boundary。需要去重的 source 必须一次只改一类，并证明它是代数重复，而不能把 selector/value 一起重定义。

### P0-3 `P3-TEMPORAL-COMMON-MODE-BYPASS`

**实现事实**

当前 P3 直接令：

```text
temporal_base = S.temporal_innovations
temporal = Linear(temporal_base + temporal_base * consequence_gate * action_gate)
```

它移除了 noisy action 对 base 的直接调制，却把几乎无时间差异的 S value 作为一个较大的加性 source 直接送入 bottom。

**实验观察**

temporal base RMS `0.516`，interaction `0.00547`，24-row variation 约 `8.4e-6`。因此该 lane 主要是公共偏置，不是可观测的时序 innovation，也没有被 W 后果实质约束。

**判断与决定**

这是“修旁路”时制造出的新旁路。先恢复 V120 P3 的 source/scale/role-bank 行为作为恢复基线。若随后要改成真正 innovation-only temporal，必须展示其输入不变量和零/非零语义；不能继续用一个大公共 base 加极小 interaction。

### P0-4 `SECOND-SIMPLIFIED-ACTION-QUERY`

**实现事实**

当前 bottom 内部仍有 V120 native physical-action lift，但 P2/P3 另用 `ActionQueryEncoder`：一个 `Linear(18,H)` 加 time MLP、sinusoidal horizon 和 learned basis identity。V120 顶层 action tokens 来自 typed physical lift：arm absolute/delta、gripper value/delta/extra、role/horizon/basis 身份共同构造。

**判断**

同一个 noisy action 在顶层和底层被两套不同几何编码。它改变 P2/P3 的条件分布，且没有独立实验依据。

**决定**

恢复/共享 V120 action-token construction；若确有必要保留轻量 query，必须先证明它与 V120 token 的尺度、字段分解和时间语义等价。

### P0-5 `GLOBAL-CLIP-STARVATION`

global clip 1.0 本身与 V120 机制相同，不是独立设计错误。但在 P0-1 的 transition preclip 极大时，它把其他所有 owner 的有效步长一起缩小。禁止用 per-owner clip、人工梯度或简单缩放掩盖；先修 P0-1/P0-2，再复查各 owner 的 pre/postclip 占比。

## 4. 已确认或高度可疑的 P1 问题

### P1-1 `FINE-CANDIDATES-COLLAPSED-BEFORE-P1`

当前 V120 progressive fine candidates 有 49 个局部候选，但 observation adapter 在进入 G 前就按 typed posterior 求期望，得到每个 local-M 的单一 slot/coordinate；P1 随后围绕该期望坐标重新取一个 3x3 microgrid。这样多峰候选在 action/object query 能选择之前已经不可逆地消失。

日志中 P1 chart posterior variation `1.21e-4`、coordinate variation `0.00238` 与这一瓶颈一致。恢复时应保留 V120 的候选轴到 P1 query-specific contraction，并沿用其 memory-safe streaming/einsum 次序；不能通过整块 materialize 换回 49 候选，也不能以缩小高分辨率读取作为代价。

### P1-2 `S-VALUES-ARE-TIME-COMMON`

当前 S 的 `interval_action_innovations` 只由 goal/history/self-block innovation 构成，object evidence只进入独立 object key/value；24 个 temporal queries 再读取这四个较公共的 interval values。`interval_variation` 还混入 learned query identity，因此不能代表 value 真有区间差异。

结果是 S 有非零 RMS，却几乎没有时间差异，P3 又放大其公共分量。恢复 V120 cumulative S 还是设计真正 data-dependent innovation 两者都有取舍；该项属于“问题已确定、修法需先展示”，不得直接再写一个新 S。

### P1-3 `EARLIER-DETAIL-IS-A-FRAME-ALIAS`

`restored_observation.py` 明确执行：

```text
previous_detail = raw_context.high_features[:, 0]
earlier_detail = previous_detail
previous_literal_rgb = raw[-2]
earlier_literal_rgb = raw[-3]
```

训练损失却把两者标成不同时间段。于是 earlier→previous feature 对比是同一 feature，而 RGB anchor 是两帧真实图像。这是明确的时间语义错误。

可选修法有两种：保留第三帧高分辨率 feature（增加显存）或取消这个虚假的 feature interval、只保留可成立的 RGB/DINO/flow 监督。具体实现需先展示显存与监督影响，不能继续复制张量冒充历史。

### P1-4 `W-IS-DROWNED-NOT-EMPTY`

W 不是“完全没接”：预测 interval variation 随 teacher 增长，batch 340 W object cosine 约 `0.73`，P2 effect 也非零。但它的幅度和区分弱于 teacher，下游又同时存在：

1. P2 consequence 消费 W；
2. static transition 直接消费同一个 W；
3. P3 common temporal base 绕过 W。

因此当前不能通过增加 W loss、route quota、hard gate 或 effect 放大来“让 W 重要”。先消除 P0 的消费者错位，再判断 W 本体是否需要修改。

### P1-5 `PUBLIC-OBSERVATION-MIX-IS-UNVALIDATED`

当前把 current content、flow-aligned visual innovation、recent motion、earlier motion 以固定方差缩放相加为一个 `public_scene_base`。它保留了三帧两段运动，方向和单位审查目前成立；但这个固定混合改变了 V120 各 source 独立参加 attention 的方式。

G 在 batch 340 的对象差异已接近 V120，因此它不是当前首要根因。暂时保留，等 P0 恢复后用 source-wise RMS/JVP 判断；若要修改，必须先展示分源输入，而不是再把它们换成另一种信息汤。

### P1-6 `S-RECOGNIZER-TARGET-GEOMETRY-DRIFT`

V120 的训练期 recognizer 用单一 whole-segment latent 监督在线 interval state。当前实现把它改成 action/state/object-key/object-value 四套 factorized latent，并把四套共同训练的 detached hidden 直接作为在线 S 目标。detach 只隔离当前 backward，不会固定 recognizer 坐标；日志又没有记录 target variation、target drift 或 online/target scale ratio。

recognizer postclip gradient 从 batch 20 约 `3.15e-3` 降到 batch 340 的 `1.45e-4`，而在线 temporal value 仍近公共。因此 recognizer 自身容易重建并不能证明 S 得到了可识别的区间意图。该改写可能合理，但没有行为验证；先以 V120 target geometry 做恢复基线，保留 factorization 前必须展示它的尺度锚定和实际区分。

### P1-7 `P1-OVERLAPPING-HISTORY-CONDITION`

当前 P1 query 等权相加 `S.temporal_queries`、S 派生的 coarse-action innovation 和 history proposal。前两项共享 S 的 goal/history 主载体，第三项再次读取 observable action history；object/detail 主要位于后续 key/value。P1 chart/coordinate variation 还随训练从约 `3.4e-4/9.9e-3` 下降到 batch 340 的 `1.21e-4/2.38e-3`。

这不是“clean query”天然有害，而是同一因果来源以多个别名重复占据 query，且缺少 V120 行为对齐。该项与 fine-candidate 提前压缩共同修复：先恢复 V120 P1 query/clean-basis 语义，再决定每个额外条件的唯一 provenance。

### P1-8 `CONTRACT-PROMOTED-HYPOTHESES-TO-INVARIANTS`

当前 compact contract 把“P3 additive temporal base”和“clean proposal - zero proposal static transition”写成 non-negotiable invariants，但二者正是未经长跑验证且已与退化对上的设计变化。契约只能描述当前实现，不能替代行为证据。见本文件 P0-1/P0-3；这两条在恢复审查结束前暂停作为设计依据。

## 5. 应保留的实现与暂不归罪项

以下内容已有源码/配置证据，不应因当前退化被整体回滚：

- V120 active `FlowDINOEvidenceEncoder`、三层 Evidence MMDiT、ordered low-rank contraction、execution value reader、event/motion heads 和 physical codec 都仍是有效主路。
- 当前 physical action codec 与 V120 正式 active mode 对齐；早期 native/decoded 指标也不支持 codec 是首因。
- 三帧 DINO/raw、两段 source-relative flow、flow-aligned history、teacher no-grad/部署隔离、真实 warp/cycle/smooth/RGB photometric anchor 的边界合理。
- optimizer 名义组别已恢复为 top `1.0x`、proposal `0.625x`、bottom `0.7x`、capacity `1.4x no-decay`；当前问题是 global clip 下的实际步长，而不是名义 LR 表。
- W teacher/effect 的 object/camera/interval 轴、G K+null、current-only deployment API 和 fresh/resume 身份检查可以保留。
- V120 正式配置中 variational CVAE posterior 和 hierarchical workspace 均未启用；不因历史名称把它们认作丢失的 active 算法。
- execution value reader 在 V120/V122 都能学习排序，但它不是充分条件，不能用它解释或修复当前 action 退化。

## 6. V120 逻辑的修改权限表

### 默认恢复/保留

- per-ODE noisy-action controlled transition；
- V120 typed physical action lift；
- decoder evidence source 数量、顺序、role bank 和真实 layer boundaries；
- P1 在 query 决策前保留 fine candidate 轴；
- active bottom、capacity、execution-value 和 action codec；
- V120 已启用且日志显示有差异的 S/W 路径，直到替代方案单独证明更好。

### 可直接修复，但必须保留审计记录

- future teacher 进入 online/deployment；
- source/target flow 单位或方向不一致；
- 名为不同时间帧却引用同一 tensor（P1-3）；
- 声明为 algebraic neutral 的 optional value 被 affine bias 重新制造；
- shape/axis 被 reduce 后再 `expand` 冒充原轴；
- checkpoint 静默加载不兼容 top。

### 修改前必须先给用户看

- learned-neutral query 改成 zero-action/zero-proposal counterfactual；
- V120 cumulative S 改成 innovation-only S；
- rollout/value source 删除、合并或清零；
- 五类 P3 source 改成三类；
- layer contract 重组；
- 把 V120 动态分支缓存成静态分支；
- 任何会明显降低显存/延迟、同时改变 active compute 的“优化”。

## 7. Schema 21 源码恢复映射

这次恢复没有继续增加新 block、gate 或 loss：

1. transition 恢复为每个 ODE step 读取当前 noisy-action token，并使用
   V120 learned neutral；缓存只保留 G3 的 512 行静态 chart。
2. action query、五类 P3 source、bottom evidence bank、source ordering 与
   layer boundary 恢复 V120 活跃语义。
3. S 恢复 V120 cumulative organizer；训练 recognizer 恢复 whole-segment
   target。history proposal 只保留辅助监督，不再冒充在线条件。
4. P1 在 query 决策前保留 49 个 progressive candidates，并用 checkpointed
   chunk contraction 控制内存；query 恢复 clean V120 action-basis provenance。
5. `earlier_detail = previous_detail` 的虚假时间帧被删除；没有真实 earlier
   high-resolution feature 时不再计算该 feature interval。
6. future row loss 仅修正一个可证明的数学错误：prediction 等于 target 时
   floored direction loss 现在可以精确为零。
7. 旧 schema 20 不能静默 exact-resume schema 21。

禁止用以下手段代替上述恢复：给 W 加权、强迫 route mass/entropy、hard gate、per-owner clip、人工梯度、缩弱 P1、恢复未启用 CVAE/workspace、增加版本号 validator、或用参数量/shape/test 数量宣称完成。

## 8. 恢复验收

### 静态验收

- 对 observation→G→S→W→P→transition→bottom 每个真实 tensor 列出 producer、consumer、动态频率、shape、单位、detach、归一化与 loss owner。
- V120 与恢复版的每个差异都必须落在本账本三类修改权限之一。
- ODE-dependent 分支不得被 online cache 静默吞掉；静态证据只构建一次，动态 action path 每 step 重算。
- bottom source bank 的 value 统计和 attention 竞争与 V120 对齐；任何 exact-zero source 有代数理由。

### 早期行为验收

- batch 20–340 action flow 不再从相近起点分叉到 V120 的 1.7 倍；
- global preclip 和 transition 占比回到与 V120 同量级，不能由单 owner 吞掉全局 clip；
- P1 query/coordinate variation、S temporal variation 不能继续接近数值零；
- W effect 的 zero/shuffle 先改变 P2/consequence，再判断 action；不以单纯非零梯度宣称接通。

### 长跑验收

- 同数据、seed、batch size 8 完成全部 8 epoch；
- 不只看 best：E1、全部 validation、E8 都与 V120 对照；
- tail/first、arm/gripper、event/motion、三 horizon bands 不能用 aggregate RMSE 掩盖；
- 若数据流和梯度几何已恢复而 action 仍无收益，再归类数据可识别性或新结构本体问题，不继续用契约补洞。

## 9. 可复现审查入口

```powershell
# 历史 schema-20 早期日志
uv run python -m clearvla.tools.audit_policy_logs `
  schema20_recovery_b8_20260811_231427.log --tail 400 --format text

# V120 完整日志
uv run python -m clearvla.tools.audit_policy_logs `
  v120_long.log --tail 400 --format text

# Schema 21 正式结果（产生后）
uv run python -m clearvla.tools.audit_policy_logs `
  runs/schema21_v120_recovery_b8 --tail 400 --format text
```
