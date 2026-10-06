# G 槽位身份、内容与共享绑定：审核注意点

> 2026-10-05。此文件是持续维护的审核与验收清单，不是新架构、不替代 `00_CURRENT_ARCHITECTURE_CONTRACT.md`。本次提交只新增本文件；不修改模型、配置、loss、探针、checkpoint、工作流或训练进程。文中的“应检查/应修”均为待办，不表示已经实现或通过。

## 1. 固定来源与结论边界

| 项目 | 身份 |
|---|---|
| 目标分支 | `codex/g-slot-identity-logit-scale-20261005` |
| 审核基线 C9 | `b229003a55b64cd220dbdfebe7ec9eb1c75d071e` |
| 基线源码树 | `275eb9cfe8856dd05fbb6815fbc22e37cbb4513a` |
| 直接父提交 C8 | `d60928853fb7b8ded17a72e0cea976c7990ff2fe` |
| 更早 repair | `d931fe01f9c6eb7d0e8f74d1ef994503020ef278` |
| 当前完整 grounding.py blob | `cc2f6ff521974dc9f96053097ebbbe9050bf776d` |

用户消息中的分支后缀 `202605` 与实际远端不符；上述分支以 HEAD 对应 b229003a 确认，不创建同名近似分支。旧 `9eed73a9` 的 run_context、R9 固定 checkpoint、正式长跑后续 checkpoint 与本次源码干预必须分别记录，不能拼成一个实验。

**现阶段结论：C8/C9是身份幅度与寻址尺度候选，不是物理对象分工、语言绑定或闭环行为的完成证明。**用户报告候选特征空间差异约 0.428、语言换目标后绑定变化仍小、address logits为零；这些是待附原始记录的本地结果，本次未读取用户权重或服务器。b229附带的是程序及提交说明，不能将它们等同于完整结果文件。

C9说明记录的 K2/K4 candidate-read cosine约0.9821→0.9632、object-chart cosine约0.99828→0.99705、content cosine约0.99971→0.99958，改善层级明显不同。0.6669→0.6670是单节点velocity的RMS，不认证动作方向、差分、24行轨迹或任务质量；instruction-pair cosine>0.99999也需同时看RMSE、符号和实际接触。[S01,S02]

## 2. 优先级总表

| 编号 | 优先级 | 注意点与完成判据 |
|---|---|---|
| A1 | P0 | C9跨整个batch求gain；同一样本更换batch伙伴的结果应被单独测试，不能用B1短测验收B8训练语义。 |
| A2 | P0 | 按来源身份区分当前G、回看G、Teacher；不能用hook列表最后一项配当前S/W。 |
| A3 | P0 | 部署no_grad缓存、合成特征平方loss、正式动作训练VJP分开；补正式目标与optimizer更新。 |
| A4 | P0 | slot、candidate read、像素后验、内容、binding、最终动作分别验收；不可跳过仍相似的object-chart。 |
| B1 | P1 | identity gain应审查空间对比、null、上下限、stop-gradient、实际有效贡献，而非只看总logit RMS。 |
| B2 | P1 | task_object_score与原object_score都含任务；查实际相对logit及训练信号，不比较权重RMS后直接加大gain。 |
| B3 | P1 | post_pool_only零address属于关闭支路，不证明共享binder无语言；开关生产与消费两端都须追踪。 |
| B4 | P1 | S的K归一化共同值、P2区间求和相消和约0.02尺度分别核查，不把std比例当作动作信息保留率。 |
| C1 | P1 | 探针修正字段归属、统计轴、静态历史说明、JSON尾部和覆盖保护，随后复跑固定面板。 |
| C2 | P1 | checkpoint/配置/源码/随机状态一致；C8/C9不是等价优化，不可改身份冒充exact resume。 |
| D1 | P2 | 保留真实小目标、完整软支持与K置换；任何新监督需有独立、置换兼容的物理来源。 |

## 3. C9新增尺度必须补的检查

当前 `_competition` 在FP32概率路径中计算：[S01]

```python
main_logit_rms = logits.float().square().mean().sqrt().detach()
identity_logit_rms = identity_logits.float().square().mean().sqrt().detach()
identity_gain = (0.25 * main_logit_rms /
                 identity_logit_rms.clamp_min(1e-6)).clamp(max=8.0)
logits = logits + identity_gain * identity_logits
```

### A1：无dim的mean把B/N/K全合并了

`logits`为 `[B,N,K]`；gain是全批一个标量。同一观测单独运行、与不同观测合批、换batch大小、分成不同microbatch，都可能得到不同身份增益和候选read。batch置换不变不等于batch组合不变；DDP每个rank的本地批次也需独立考虑。这是新增的跨样本数值依赖，不是参数新增问题。

本次从blob核验的完整文件提取真实 `_competition` 及其helper，用CPU/H32/K4/N24、三个新参数种子905/906/907做了局部检查：锚样本完全相同，更换同批伙伴后read最大差约0.00175—0.00370；合批复制锚样本的对照最大差不超过2.24e-8。**未执行GRU/G3、真实checkpoint或完整策略；这些数值不等于R9或生产误差。**它只确认该代码路径的batch耦合可实际出现。

待办：固定同一模型与观测A，对比 A、AA、AB、ABC、分块/整批，记录每样本gain、owner/null、read、G值和最终命令。若目标语义要求样本独立，后续补丁应在同一真实样本的合法来源上定义尺度；不能未经数值与行为对照直接宣布按样本RMS已修好。

### B1：总RMS不是空间分工所需的对比量

两个K若在全部候选上的logit只差常数，第二次逐K读归一化会消掉该差异。设动态key为q、identity为e、候选为c，则应看：

\[
\operatorname{center}_n\log(R_{kn}/R_{jn})
=\operatorname{center}_n\{[(q_k-q_j)+g(e_k-e_j)]^Tc_n/\sqrt H\}.
\]

中心化和方差使用同一合法候选权重。分别记录动态项、identity项、合成项的空间方差与协方差；总logit大、identity占比25%均不保证这些空间log-odds有用。共同场景偏置可抬高main RMS，却不增加对象间位置区分。

边界必须记录：raw/scaled identity RMS、gain、是否触顶8、是否触及1e-6下限、real/null质量、小而合法的支持。未触下限时，实际scaled/main比例为 `min(0.25, 8*raw_identity/main)`，不是永远25%；main为零时当前gain也为零。增加相同的零填充在无下限效应时可同时缩放两个RMS而抵消，不应未经测试声称任何padding都会改变gain。

缩放只作用于真实K logits，null logit未同步改变；即便K+null轴和归一化公式没改，**real-versus-null事件概率仍可改变**。保留质量守恒、全无效、非有限隔离、小目标和极低真实质量测试。不能用predicted confidence/allocation取代物理支持，也不能强制每个K等质量。

两个RMS均detach：反向忽略gain对输入的导数，但直接logit路径仍有梯度。将其标记为明确的stop-gradient尺度策略，分别测试实现VJP及重新计算gain的有限差分；不能要求二者自然一致，也不能称为原计算的等价性能优化。

## 4. C8遗留的状态语义仍须保留

父版本更新为 `h'=u+0.5*stopgrad(max(||u||,1e-6))*e`，u是GRU与FFN之和，e为seed中心化后逐slot单位化方向。[S01,S02]

身份范数跟随u不保证其投影到候选变化方向；slot cosine下降不证明read或像素后验分开。前向完整导数含 `0.5*e*u^T/||u||`，实现detach省略这一项；不能只用非None梯度验收。

seed减K均值后再逐slot单位化、乘各自不同尺度，最终identity carrier不保证K均值为零。应记录实际carrier的公共分量，不能沿用“中心化所以不增加共同值”的绝对保证。重新中心化、删除detach或增加identity常数都是新行为修改，不是文档修正。

首轮须分别读取 initial slot、GRU输出、加FFN、加identity，再在固定candidate/prior/validity下送回原竞争函数。对每一步同时看空间log-odds、read重叠与物理区域覆盖，区分原始差异被缩小、共同分量增大以及LayerNorm后差异减弱。

低联合mass不等于低强度更新：GRU读取逐K条件期望，并未再乘总allocation。existence是对象相对null的置信，validity是观察支持，都不证明相对其他K有独占物理解释。木桌本来可见，不会被这两项自动判为错误；直接乘raw mass则可能伤害小物体。

## 5. 已有及新增探针：先修测量再据此归因

| 脚本/模式 | 当前限制 | 后续必须做的修正 |
|---|---|---|
| `probe_g_slot_identity_short.py`的`current_calls[-1]`；dataflow及`probe_second_plan_binding.py`的`facts[-1]` | 同一encode随后可能回看过去观测，再调用同一grounder；最后G可能是过去源 | 与当前`training_state.top.facts`或当前cache.content精确身份匹配，并记录source role、time和encode序号；不用allclose判断来源。 |
| `probe_g_competition_logit_scale.py`逐次记录 | 未给各次竞争绑定当前/回看作用域和iteration；记录raw比例，没有完整记录实际gain/scaled结果 | 分作用域记录三轮及G3前竞争，附B/N/K形状、支持、cap/floor和有效空间对比。 |
| 部署cache后反传 | `deployment_cache`内的no_grad已截断G/S生产图 | 分开部署前向与允许梯度的真实online encode；不得给detached cache加requires_grad冒充恢复生产者链。 |
| 独立gradient脚本的content/S/W平方和 | 合成特征目标，不是正式动作/重建训练，也未证明动态P2/整轮采样的梯度作用 | 准确保留其局部范围；补原训练loss分项、正式总loss、参数owner、原裁剪顺序和实际一步更新。 |
| “209参数通过” | 统计非None梯度的参数张量条目，不保证非零、方向有效或优化器更新 | 分别导出None/zero/finite、分解前激活VJP、参数VJP和参数更新量；不把共模抵消误判断图。 |
| 单帧`_history`再伪装四步 | 同RGB/state/命令重复，时间改为4并标observed，仅是静态接口fixture | 静态G审计可保留且明确命名；反馈、恢复与时序结论必须使用真实源历史，禁止虚构observed。 |
| 字段归属 | validity曾命名为mass；predicted_dynamics曾命名P2 | joint assignment/conditional read/existence/validity分别记录；W预测与实际P2 terminal输出分开。 |
| 统计轴 | 残差整体RMS与逐通道interval std平均不同 | 输出定义、维度、单位和分母；K、I、C、空间和动作行不可混比。 |
| JSON与复现 | 新旧若干脚本追加字面反斜线+n（`'\\n'`），不是换行；硬编码路径且可覆盖 | 改为真实换行，落盘后标准json回读；参数化checkpoint/input/device/output；默认拒绝覆盖。 |

这些限制不能推出所有短测数值都是假的；它们意味着当前脚本不足以证明某些来源和因果归属。旧报告中的误测应标为范围受限而非删除。`probe_second_plan_binding.py`的`cos(v)`辅助函数比较自身，仅为自相似且当前未调用，不得以后拿它作跨指令相似度。

本分支不包含已复核的完整本地GPU结果文件。下一批应单独交付小型JSON摘要及其hash；不要将权重、完整hidden、原始日志或大tensor写入架构文档。诊断同步开销不能用于性能测量。

## 6. 0.428为什么不能跳过G的中间证据

新binding探针的 `g_candidate_cell_variation` 将 `candidate_content`展平所有候选轴后求std再平均。它不是K2/K4输出差异，也不是独立物理区域覆盖。[S03]

必须分解以下三个映射：

\[
h_k\longrightarrow R_{kn}\longrightarrow
P_{kx}=\sum_nR_{kn}Q_{nx}\longrightarrow
f_k=\sum_nR_{kn}v_n+W_{res}h_k.
\]

Q表示局部候选对当前图像的来源映射，具体支持/概率归一化以生产代码为准。不同候选编号可映射同一像素区域；不同区域也可有相似材质和DINO共同成分。最后内容差为 `(R_k-R_j)(V-mean(V)) + W_res(h_k-h_j)`；read与value方向不匹配、期望相消、或两项抵消都可能使content继续近似。[S01]

依次记录复合candidate key、真实candidate_content、read、每相机current_image_measure、aggregated_content、slot residual、二者和，以及semantic/appearance/geometry各自的重加权read。共同typed先验可把不同parent read重新拉向同一候选，不能假定四条属性自动沿同一分工改善。

至少包含“固定V换R”“固定R换V”“聚合项/残差分别中和”的局部诊断。干预须保持支持、来源与后续缓存一致；完整策略中不能静默只替换一个字段。相似背景本身允许，目标物块是否至少被某个K可靠覆盖才是任务相关验收；禁止为降低cosine给同一区域加不同内部ID。

## 7. 追task_object_score时必须按真实调用链

当前binder直接消费 `protected_goal` 和 `objects`，发生在 `interval_goal`读取之前；objects已包含content及semantic/appearance/geometry投影。不能用下游interval-goal attention不变解释上游binding不变，也不能称当前binder在softmax前完全没见颜色/语义。[S04,S05]

两条评分分别为：

```text
object_score = score(task_read(query(obj,history), task) * tanh(compatibility(obj)))
task_object_score = Linear([obj*task_context, obj*tanh(task_context)])
```

原object_score也有语言，不是纯视觉；新增项已有真实乘性交互，不能再套用旧 `Linear([obj,task])` 的可分离性结论。权重RMS、置零整支路的输出RMSE不是可加的视觉/语言贡献百分比。[S04]

同一画面、状态、历史和噪声，只换颜色、保持方向不变，分别记录两项与总项的每K logits及：

\[
\Delta_{kj}=[\ell_k(t_a)-\ell_j(t_a)]-[\ell_k(t_b)-\ell_j(t_b)].
\]

再记录真实K条件分布、real/null总质量、目标物理区域质量及实际动作。这样区分公共偏置、真实相对排名、相互抵消和下游未消费。换方向实验单列：同一目标左/右动作不同本来合理，不能要求全部语言变化都必须改变K。

训练检查顺序：`requires_grad`与唯一optimizer owner/真实lr → 正式loss对binding相对logit的VJP → 两条评分的激活与参数VJP → 梯度裁剪前后 → 实际optimizer delta → 固定颜色面板的前向贡献。特征平方和、参数count、attention argmax相同均不是此链验收。

## 8. 零address必须区分“未选中”和“选中但失效”

S构造器仅在 `target_object_address_mode=shared_target_prior_v1`时创建 `_ZeroStartTargetAddress`；否则模块为None，`_typed_relevance`显式输出零address。P2在post_pool_only下也将address条件置零。[S05,S06]

因此，先核对checkpoint序列化的选择、实际实例类型/参数owner，以及S生产和P2消费两端；零address在关闭模式下不是训练失败。该后置address不是已先算出的binder的输入，不能把二者混称为同一条task→K路径。

shared binding模式下semantic K后验直接沿用binding；geometry只在给定K内部选相机。若新address仅给同一K各相机加同一个常数，softmax会消去。不能只打开一个开关或新增未消费参数就宣布恢复任务寻址；改动仍需维持唯一共享K+null所有权，不给P1/P2另造选择器。

没有固定K标签不自动等于bug。若真实目标没有被任何K有效覆盖，binding重加权无法凭空恢复它；先测物理区域覆盖上限。若考虑辅助监督，只能用独立可靠、K置换兼容的区域/对应来源，不以“动得最多的块”或当前G自身作为无条件真值，不直接固化K编号、颜色或等分配。

## 9. S/P2保留的串联风险

S新增任务值中存在 `b_k / sum_j(b_j)`乘同一个U_i后对K求和。在相应typed来源全有效、真实质量大于下限且固定上游query时，增量严格回到U_i；真实质量很小也可被重新归一化到完整幅度。原G事实项和上游binding依赖仍在，不能说整个S与对象无关，但新增std可能主要反映共同任务/区间模式。[S05]

P2的target尺度为 `0.02 + 0.02*tanh(parameter)`；名称含zero_start不代表有效初值为零。`source_interval_variation`和`target_value_interval_variation`都在区间汇总前，后者本来就是source乘scale。约1.97%的比值不是最终信息保留率。[S06]

必须记录真正的 `scale * sum_i(p_i U_i)`，其共同/残差、与基础效果的夹角、再到P3/最终命令的敏感性。若U_i含零均值区间残差，均匀p使其相消；放大scale不能解决严格相消。高interval cosine也不等于值幅度相同，要看中心化范数与奇异值。

S common/residual精确重组时common节点梯度可抵消，应监控分解前被消费的V；公共载体去通道均值仍大，不排除固定interval模板。C5小gain在BF16的`1+modulation`可被舍入，需检查真实forward增量，而非仅看参数更新。既有的C3/C5/C6、global和CT是不同替代路径，需逐条标清，不能一条静默就称全链断开。

## 10. 下一批最小可复核面板

| 面板 | 固定项/干预 | 关键输出 |
|---|---|---|
| 当前来源资格 | 当前/回看作用域；静态fixture和真实历史分开 | 对应facts身份、源时间、支持、命令边界；严禁串源。 |
| C8/C9配对 | 同一权重/input/noise；父/候选；B1、复制样本、混合伙伴/分批 | gain、cap/floor、当前每轮log-odds、owner/null、最终像素分布。 |
| G空间到内容 | 固定R换V、固定V换R；聚合/残差拆分 | 每相机目标/木桌覆盖、重叠、内容差异与共同模式。 |
| 语言到目标 | 同画面同方向换颜色；同色换方向作为另一组 | 两项score、真实K相对log-odds、目标区域质量。 |
| 目标到动作 | 固定其他源，干预shared binding并重算全部消费者 | S/P1/P2实际值增量、coarse、proposal/W/refined、最终24行与实际前8行。 |
| 正式训练路径 | 可微online encode、真实监督和原loss；单独一步更新 | 参数归属、finite/zero/None、裁剪、delta及真实前向贡献。 |

每条记录最少带源码提交/dirty文件hash、checkpoint hash或既有强身份、checkpoint实际配置、图ABI、state/action normalizer、DINO身份与精度、完整输入来源、当前/回看标记、RNG状态、dtype、B/N/K/C/I轴、计算模式、执行行数。服务器路径与文件名不是已读取内容；R9与正式长跑必须分开。

固定物块小面板与留出的新布局都要覆盖；slot编号不能永久映射颜色，双相机坐标未经标定不得平均成世界点。接受概率的误差阈值应先由原版重复性建立，不为了候选通过修改标准。局部velocity范数不代替完整命令、夹爪边界或闭环阶段。

## 11. 改动顺序与不可越过的边界

1. **先修探针与来源/序列化，补A1 batch独立性对照。**保留原失败和中断证据；不把不可信字段继续用于下一轮归因。
2. **再区分G的candidate→pixel→content映射。**真实目标覆盖不足时不要直接跳到绑定loss；不能通过排除木桌像素、扩大slot residual或正交化编码伪造对象分工。
3. **在已有目标覆盖下追binder真实训练和颜色相对排名。**只改一个已确认节点，保持同一共享K+null，不新增外接selector、不硬化argmax。
4. **再检查对象相关值到S/P2/最终动作。**保留必要的公共任务/方向信息，但不要让归一化共同gate冒充对象特异性。
5. **正式训练与部署验收单列。**C8/C9改变了前向与停止梯度策略；同shape旧权重仍可用于明确标注的诊断初始化，不得改checkpoint身份冒充精确恢复。先查真实加载验证覆盖哪些文件/模式，不能假定新增源码自动被全部身份规则捕获。

保持完整候选、相机、软分布、来源支持和K置换；不改变动作归一化、原生控制语义、R1/R8或ODE计划来掩盖问题。不得热改正在运行的工作树、覆盖权重/结果、强推或更改master。此文档没有修改detach、gain、mask、模式或loss，也不授权启动训练。

历史问题不得无证据搬到当前：9eed终点来源缺失只在那份运行确认；617的P1缩放回归不能自动当作当前bug；C7在兄弟分支完成不代表本checkpoint启用。Teacher自引用/未知与零变化、R1/R8、clean-arm/sampled-arm端点、W(proposal)/最终动作条件、物理穿透与控制器停滞仍应按实际图和阶段分别验证。已有层的有限梯度、loss下降或总成功率不能宣布这些全部闭合。

## 12. 源码索引与本次交付范围

以下索引均以首节的b229完整提交为准，后续变更应更新本文件结论，不默默沿用移动HEAD。

| 索引 | 固定源码位置/核对内容 |
|---|---|
| S01 | `clearvla/mainline/model/grounding.py`：`_competition`约395—487、G迭代更新、typed_reweight、aggregate、camera_aggregate与当前图像pushforward；完整blob已核验。 |
| S02 | `docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`：C8/C9尾部记录；统计为提交者报告，非本次checkpoint重跑。 |
| S03 | `probes/probe_g_competition_logit_scale.py`、`probe_second_plan_binding.py`及继承的三份`probe_g_slot_identity_*.py`；目录/源码与父报告逐项核对。 |
| S04 | `clearvla/mainline/model/task_execution.py:531—618`：TaskConditionedTargetBinder的两个任务相关分数与唯一K+null。 |
| S05 | `clearvla/mainline/model/intent.py:430—447, 575—750`及forward中的binder→interval_goal顺序；当前blob `3e82e7712d3c47ccf1f88480060bf6a5d8db8378`与此前已读版本一致。 |
| S06 | `clearvla/mainline/model/compiler.py:740—845, 1080—1426`：address选择、空间/区间消费者和target-value指标；blob `2954020ab5f2ecaf6b2ffcc30c12834347903b62`。 |
| S07 | `clearvla/mainline/model/policy.py::_replay_executed_world`、`clearvla/mainline/runtime/sampling.py::deployment_cache`、`scripts/probe_calvin_internal_layers_npz.py::_history`；当前/过去与无梯度边界。 |

本次完成：分支/父提交/变更核对，完整grounding blob核验，继承报告与当前源码对照，三种合成batch组合的真实竞争方法检查，以及文档检查。未完成也未宣称：R9/正式checkpoint重放、GPU训练、整网梯度/闭环或全仓库测试。此文档只提交决策统计、来源和验收要求，不复制权重、tensor cache、原始大日志或服务器数据。

## 13. 2026-10-05：完整目标交换的下游因果复核

本节记录的是当前分支短 checkpoint 的只读测量，不是生产修改或正式训练验收。测量使用 `probes/probe_target_binding_measurement_v2.py`，固定同一观测、同一噪声，并交换最远的两个有物理支持的 K 槽位；null 质量、真实质量多重集和支持均保持不变。新增提交 `1ea28eaf` 记录 proposal/refined 每次 velocity 调用，`8c092dbd` 增加受控的 P3、底层 P3、protected detail、P1 precision 缩放。短 checkpoint 身份为 `short-bs8-gpu3-r1/checkpoints/best.pt`，`global_step=64`，因此下列数值不能外推为训练完成结果。

### 13.1 轨迹复核结果

四条指令的 full proposal→W rebuild→refined 目标交换，平均末端差异为：P3 temporal `.0328`、底层 P3 lane 0 `.0235`、lane 1 `.0061`、P3 bridge `.0166`、protected detail `.0464`、最终 native action `.0567`。逐次 ODE 调用中，P3/bridge 的差异随时间从约 `.0045` 增长到 `.0166`，而 protected detail 一直约 `.0465`；这表示 P3 路径确实被消费，但当前短 checkpoint 的最终目标差异主要由并行的 protected-detail 载体提供。

### 13.2 受控缩放的因果证据

| 只读干预 | P3 temporal 交换差异 | P3 bridge 交换差异 | 最终 action 交换差异 |
|---|---:|---:|---:|
| baseline | `.0324` | `.0165` | `.0564` |
| coordinator semantic 输入置零 | `.0249` | `.0147` | `.0550` |
| 底层 P3 两个 lane 置零 | `.0293` | `.0105` | `.0578` |
| protected detail 置零 | `.0243` | `.0115` | `.0126` |

protected detail 置零会使最终目标差异显著下降，而 P3 置零只改变其中一小部分；这不是“P3断路”的证据，反而说明继续只放大公共 P3 载体会掩盖真正的并行消费关系。`p1-precision=0` 的最终差异会上升到约 `.138`，属于移除相消路径后的非线性重排，不能当作“precision贡献为负”的线性分解。

源码对应关系是：P1 `build_static` 将 target detail 加入 factual protected detail；P2 consequence 在此基础上加入 typed effect/interaction；P3 另外生成 temporal/state-change 两个可选 lane；bottom 在 `_read_policy_delta_bank` 中把可选 P3 与两个 protected carrier 分开读取。因此，下一步应先在同一完整 checkpoint 上复测这条分叉，而不是修改 bridge 固定缩放。

### 13.3 正式长跑状态与下一门槛

正式长跑仍使用启动时锁定的源码 `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，没有被上述探针提交改写；截至 batch `6300/11012`，运行约 `6.41 s/batch`，P2 target interval variation `.1642`、source variation `.3327`、P3 temporal RMS `.3393`、P3 routed update `.4.236`、fixed bridge scale `.25`、null mass `.0015`、source effective count `3.98`、capacity `.9995`、execution gate `1.0`。日志审计仍为 JSON/traceback 完整，只有预期的 capacity-saturated warning。正式最终 checkpoint 和同 branch target-swap 尚未产生；完成前不据此改生产网络。

## 14. 2026-10-06：protected-detail 实际生产者六源账本

本节承接 13 节的短 checkpoint 复核。报告中的关键修正已落实到探针：`protected-detail` 不是纯视觉细节，而是 P1 当前 factual lattice、P1 target read、P2 semantic effect、P2 geometry effect、P2 semantic interaction、P2 geometry interaction 六项在 P1/P2 边界合成后的 carrier。探针只读包裹真实 forward，不改变生产模型、梯度或正式长跑。

### 14.1 生产路径和闭合

- P1 `build_static` 的 factual carrier 按真实调用顺序拆为 `(updated-clean)` 与 `target_detail`；四条指令 target-swap 的 factual 闭合误差为 `1.38–1.92e-9`。
- P2 consequence 使用源码中的分组顺序 `factual_base + (semantic+geometry) + (interaction.semantic+interaction.geometry)`；闭合误差为 `0`。
- bottom `protected_detail_basis_attnres` 的 route probability 与 value contract 从同一次 reader 输入重算；重算与真实 reader 输出误差为 `0`。
- 六个来源在进入 reader 前的 carrier 重组误差为 `0`，因此没有发现来源截断、错配或漏分配。

reader 输入是先把已经合成的 carrier 转成 action-query dtype，再进入 bf16 路由归约。若把六个上游张量分别转型后再相加，会出现可重复的 dtype/归约残差：carrier RMS 约 `2.0e-4`，路由后的 residual update RMS 约 `1.9e-4`，六源账本与真实输出的未校正差约 `0.9e-3`。这属于转换和归约顺序的数值误差，不是网络路径丢失；探针已单独记录 `ledger_cast_residual`，不能把它归因给 P3 或 bridge。

### 14.2 目标交换的当前读数

v11 在同一观测、同一噪声和同一 checkpoint 上覆盖四条颜色/方向指令；每条 baseline/swap 的事件轨迹均已写入。最终 native action 的 arm RMSE 为 `.0580–.0621`，gripper RMSE 为 `0`。这些仍来自 `global_step=64` 的短 checkpoint，只说明链路可测和下游消费存在，不能代替完整训练闭环。

### 14.3 正式长跑状态

正式训练仍锁定启动提交 `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，探针提交没有改写它。最新日志到 batch `8000/11012`，最近窗口约 `6.16 s/batch`；loss、ledger gap（约 `1e-8`）、梯度和数值项均有限，没有 traceback、OOM、NaN 或异常梯度尖峰。P2 target interval variation `.1717`、P3 temporal RMS `.3429`、P3 state-change RMS `.2852`、bottom capacity `.9997`，仍属于运行中的健康窗口；最终 checkpoint 尚未生成。

下一步是在同一分支的最终 checkpoint 上重跑这组六源账本，并把 arm/gripper、P3 optional、protected carrier 三条消费链分别与完整闭环结果对齐。当前证据不足以修改 bridge scale 或 P3 拓扑。

### 14.4 六个来源的目标交换贡献

以四条指令的同观测 target-swap 为例，按真实 bottom route 的 `beta*lambda` 逐源重算（数值为底层动作 hidden 更新的 RMSE，不是 6 维原生命令或 TCP 位移，不能线性相加成“贡献百分比”）：

| 来源 | RMSE 范围 |
|---|---:|
| P1 factual lattice `(updated-clean)` | `.000031–.000033` |
| P1 target read | `.02081–.02202` |
| P2 semantic effect | `.03651–.03921` |
| P2 geometry effect | `.00584–.00634` |
| P2 semantic interaction | `.00577–.00622` |
| P2 geometry interaction | `.00070–.00075` |

P1 target 与 P2 semantic 是当前 protected-detail 目标差异的主要两条来源；factual lattice 几乎不随这组颜色/方向交换变化。P2 geometry 和 interaction 仍有可测的下游读出，但量级较小。由于这些向量会相互抵消，RMSE 仅用于定位来源，不能把它们当作独立百分比。P1 target/factual 与 W、S 形成的 semantic 在 consequence 汇合，P1 还影响查询和交互；二者不是同一信号的连续测点，不能把其 RMSE 比值称为串联放大率。下一节区分实际生产值变化、bottom 读取系数变化和 P2 内部来源；仍不支持直接改 bottom bridge 或公共载体。

## 15. 2026-10-06：八源双状态分解、W 重读与单项 baseline 钳制

本节是在源码 `785627e9b1d875bb50b7e751f23ce479afe39845` 上新增评估探针的结果，生产网络没有修改。输入仍为 `short-bs8-gpu3-r1/checkpoints/best.pt`（step 64）、同一 standard 观测、四条蓝右/红右/粉右/蓝左指令和固定噪声 12345。交换的是有支持的 K 权重；继承探针的坐标选择不构成真实颜色物块的身份认证。每条重复两次 baseline，记录 proposal/refined 的调用 0、5；调用 5 是终点 head，不是额外积分步。

### 15.1 计量契约

`probes/probe_source_delta_consumer.py` 从真实 producer/reader 返回帧获取 W 读取、S 目标补充、W×S 调制、原始共同 P2 contract、实际 bottom probability/value scale 和 reader 输出。semantic 三项共用原始 semantic+geometry contract；八个来源共用完整 carrier 的实际 bottom 系数。新观察器不另算 softmax，不给每个来源单独收缩。

`probes/source_delta_ledger.py` 离线以 float64 计算 `D_j=sum_q(beta*lambda*X_j)` 的对称双状态恒等式：

```text
delta D_j = sum(mean(beta*lambda)*delta X_j)
          + sum(mean(lambda)*delta beta*mean(X_j))
          + sum(mean(beta)*delta lambda*mean(X_j))
```

它是数值分账，不能解释成独立因果贡献百分比。NPZ 保存 source/checkpoint/input/probe 身份及 stage、integration/call、dtype、轴；不从 JSON 摘要反推张量。实际 reader 与 float64 分账的数值残差，以及残差在两状态间的变化，单独保存。

### 15.2 四指令、16 对节点的结果

| 项目 | RMSE 范围（bottom hidden 更新） |
|---|---:|
| 实际 reader 的目标交换差异 | `.04339–.04740` |
| 源值变化项之和 | `.04359–.04755` |
| basis 概率变化项之和 | `2.37e-7–3.86e-7` |
| value 收缩变化项之和 | `.000329–.000366` |
| 两状态数值残差之差 | `.000934–.001018` |
| W 读取项的路由后差异 | `.03170–.03482` |
| S 目标补充的路由后差异 | `.01817–.01986` |
| W×S 调制的路由后差异 | `.0000380–.0000413` |

包含独立数值残差后，float64 代数闭合最大绝对误差 `8.19e-16`；这不表示生产 bf16 的残差为零。P2 semantic 三项的转型/加法残差约 `.000122–.000207`。上述实际窗口以源值变化为主，不能把报告中的合成路由案例直接当作本 checkpoint 的主因。

重复性也不是全部严格为零：蓝右/蓝左 baseline 重复一致，红右/粉右的 7 维命令重复 RMSE 分别为 `.000841/.000676`，reader 重复差异最高 `.000549`。因此不能把邻近此量级的小项当成确定机制。

### 15.3 W 预测变化和读取上下文的区分

局部 P2 进行 W0/W1 与 read-context0/1 的 2×2 配对。每份 candidate world 始终携带自己的原 action condition 并通过原身份验证；不重新标记 cache。read-context 同时包含 P1 query 和 S，这一步尚未拆开两者。

W semantic 原值差异 `.03189–.03502` 中，对称 read-context 项为 `.03189–.03492`，candidate-world 项为 `.000430–.000485`。固定 baseline W、仅换读取上下文，差异仍为 `.03187–.03492`，方向与真实差异 cosine 至少 `.99985`。这些是 **P2 共同 contract 前的 hidden 值**，与 15.2 的 bottom 单位不同。候选 W 变化项接近重复数值底噪；不能据此宣称 W 没用或未来预测正确。

### 15.4 只钳制定位到的 W 读取项

蓝右完整 proposal→W rebuild→refined 评估中，在全部 12 个对应节点把 raw W 读取项换回该节点 baseline 值。S 补充、调制、geometry、joint contract 及全部后续消费者正常重算；没有整条 lane 置零。

| 原生命令差异，相对 baseline | 正常 binding swap | swap + W 读取钳制 |
|---|---:|---:|
| arm 全 24 行 RMSE | `.06237` | `.03637` |
| arm 实际执行前 8 行 RMSE | `.03422` | `.02090` |
| gripper RMSE | `0` | `0` |

该干预确认 W 读取值确实影响最终 arm；不能把非线性差异的下降率称为独立贡献比例，也不证明命令方向正确。

### 15.5 自然语言选择仍未验收

同一观测下，相对蓝右，红右/粉右的自然 binding RMSE 仅 `.000809/.000253`，arm 前 8 行差异 `.001499/.001121`；蓝左的 binding RMSE `.007583`、arm 前 8 行差异 `.003075`。而人工 binding swap 的 arm 前 8 行为 `.03212–.03438`。自然颜色响应明显更弱，部分读数接近重复底噪；不能用人工交换有效替代自然目标选择正确，也不能从 step 64 宣判训练完成后的能力。

下一步使用同分支最终 checkpoint 复测相同分解和单项钳制，并与固定 18 例、6 任务×3、seed 0、max_steps 360、execute_rows 8 的闭环逐阶段对齐。若自然颜色重新指向仍弱，先核对真实物块覆盖、binder 的相对排名和读取内容，再判断是上游选择还是下游执行；当前没有依据扩大 gain、添加差异 loss 或改 P3/固定 bridge。

### 15.6 复现与保存位置

结果根目录为 `/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/`：`source-delta-short-all4-world-v1` 保存 16 对 NPZ、完整原生命令及 summary；`source-delta-short-clamp-w-v1` 保存单项钳制；`source-delta-short-combined-check-v1` 用于组合选项回归。初次 wrapper 识别失败的 `source-delta-short-v1` 保留为失败记录。

在当前 checkout，使用模型环境 Python、`PYTHONPATH` 指向 checkout、`CUDA_VISIBLE_DEVICES=3`、HF/Transformers 离线模式运行：

```bash
python -B probes/probe_source_delta_consumer.py \
  --checkpoint /data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/short-bs8-gpu3-r1/checkpoints/best.pt \
  --t5-condition /data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt \
  --observation-dir /data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003 \
  --output-dir /absolute/new/result-directory --repeats 2 \
  --cross-world --clamp-source e_semantic_w
python -B probes/source_delta_ledger.py baseline.npz swapped.npz --output paired_result.json
```

输出目录必须不存在，禁止覆盖历史结果。离线固定源/换 beta、固定 beta/换源、同号/反号残差的代数检查通过；实际 16 对输入通过轴、有限性和身份对齐检查。文件 hash 写在各 NPZ 的 metadata 中。仓库只保存探针和决策统计，不提交 checkpoint 或 NPZ。

最终权重的独立接续脚本为 `/home/sen.wang/mysh/clearvla-gslot-source-delta-final-20261006.sh PROBE_COMMIT`，只接受本次长跑的 epoch 1 / step 11012 / source `a2d597d...`。它检查生产及探针源码未漂移，等待 GPU3 空余至少 16 GB，再执行上述四指令、重复性、W/read-context 配对与 W 单项钳制。任务状态与脚本快照保存在 `probes/source-delta-formal-final-20261006-r1-job/`；与已有 GPU4 的 R8 闭环接续任务相互独立，不重启训练。

## 16. 2026-10-06：K 坐标含义与 EMA 的实际状态

当前配置为 `entity_chart_mode=current_image_support_v1`。`grounding.py:929–935` 将完整候选位置分布推到当前相机图像，`vision/entity_chart.py:215–220` 在每个相机内归一化后求二维一阶矩，输出 `[B,K,C,2]` 的 normalized xy，`align_corners` 对应 RGB 两端像素中心。它是读分布重心，不是经标定的 3-D 位置、语义物体标签或跨时跟踪的身份保证。生产保留 C 轴；旧探针 `_select_physical_pair` 对双相机 xy 加权平均仅能定义其人工交换，不能作为真实物块间距证据。

step 64、同一 standard 观测的逐相机复测：坐标与实际分布重新求矩的最大误差为 `0`。但 200×200 上方相机内 K1 中心约 `(104,117)`、空间标准差约 `(61,36)` px；K2/K4 中心约 `(92,7)/(101,6)`，主要读背板上沿；K3 中心约 `(27,183)`，主要读桌腿/地板附近。K2/K4 的条件分布重叠质量 `sum(min(p,q))` 在上方/手腕相机为 `.9095/.8435`。因此本样本有明显背景和重复区域读取，不能把“求矩正确”当作“物块定位正确”，也不能据此外推正式最终权重。没有物体分割/标定真值，本次不报告伪精确的物块定位误差。

复测目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/k-coordinate-ema-short-20261006-v1/`。`summary.json` 保存 checkpoint/input/source hash 和统计；`per_camera_k_coordinates_aligned.png` 的热图节点与 RGB align-corners 端点对齐；初稿热图使用普通 imshow cell extent，最终分析只使用 corrected aligned 图。每幅热图独立缩放，不按颜色亮度比较不同 K 的总质量。

当前主线 Teacher 是 `training_targets.teacher: ObjectFutureTeacher`，其 `semantic_content_key`、`appearance_content_key` 均为冻结的 `[64,768]` 正交初始化投影，实际 current-reference mode 为 `g_assignment_v1`。`observation_association.py:76–88` 冻结参数，`:342–389` 仍用当前 G 坐标构建几何匹配先验；冻结参数不代表标签独立于 G。`training/engine.py:1187–1194` 没有 EMA 更新。旧 observation encoder 虽仍实例化 decay `.995` 的 Teacher-G 投影，但其更新只位于旧 `teacher_interval_targets` 路径，从当前 adapter 调用的 encoder 方法不可达；`teacher_supports` 只调用冻结 DINO 内容变换。

指数移动平均 Teacher/权重与 Efficient Multi-Scale Attention 是不同方案，需分清用户含义。前者可作为训练目标/权重稳定性的候选，无法凭平均生成正确的对象定位或语言标签；直接平滑 K 坐标还需先解决跨时 K 身份和相机运动。后者是视觉特征注意力，需要另行定位特征分辨率/可分性不足的证据，不能由现有 EMA Teacher 代码推断效果。本次只检查源码和短 checkpoint，没有改网络、监督或现有长跑/接续探针。

## 17. 2026-10-06：完整权重复测已接续，尚未得到长跑结果

截至 12:48 UTC，`train-bs8-gpu0-r1` 的最近日志为 batch 10300/11012，窗口耗时 6.351 s/batch；最终 checkpoint 目录仍为空。这里不把 step 64 探针结果写成完整训练结论。

补齐 K 的物体覆盖测量：复用 `scripts/export_calvin_probe_observations.py:state_for_initial_condition` 的原观测生成方式，四种布局的 top/wrist RGB 均与已保存 NPZ **逐像素完全一致**。直接导出模拟器 body segmentation，按真实物块 UID 取得红/蓝/粉可见区域。粉色块在这四种布局中均不可见，不能把零可见覆盖判作定位失败。最初按官方初始状态函数重放的图像不匹配，v2 结果被拒绝；只使用精确匹配的 v3 masks。

新测量从真实 G producer 的 `camera_read` 与 `CurrentImageSupport` 读取完整 M/N 分布，在实际采样坐标上双线性读取物块 mask，再按生产相机条件概率积分。没有用 8×8 热图格心或 K 重心估算物体覆盖。standard / step 64 的脚本校验通过：完整位置支持重算中心与生产 K 坐标最大误差 `3.84e-7`，每相机概率预算误差小于 `3e-7`。它仍只是测量校验，不代表最终权重的定位结果；此计量评估的是可见图像读质量，并非 3-D 误差或跨时身份。

新增一次性接续任务：

- helper：`/home/sen.wang/mysh/clearvla-gslot-final-k-p3-20261006.sh PROBE_COMMIT COORDINATE_PROBE_SHA256`；
- job：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/final-k-p3-20261006-r1-job/`；
- 固定探针源码 `6fdb2eb63935c3244635ee6e0f71c7fb8c775e8a`，coordinate probe SHA-256 `7ccf1e9faaec96233bdec875ed3d9dd7cc2af8f9f7cfdf3c2091aed1ec246484`；
- 只接受 epoch 1 / step 11012 / training source `a2d597d27e001d3bbc901f133d25d6ca5a4cf6a6`，保存 checkpoint hash；等待既有 source-delta job 结束及 GPU3 至少 16 GB 空闲；
- 四布局逐相机 K/物块覆盖，然后四指令、重复 2 次、完整 proposal→W rebuild→refined 的 baseline、P3 semantic zero、bottom P3 zero、protected-detail zero；
- 保留 15.6 的八源/W 单项钳制和 GPU4 上既有的标准闭环（18 例、6 任务×3、seed 0、max_steps 360、execute_rows 8）。

helper 的 shell/嵌入 Python 语法、当前源码 guard 和短 checkpoint 的实际测量均已验证；任务已启动等待最终权重，尚未完成这些 full-checkpoint 测试。每阶段失败独立记账，不能因部分成功宣告全部通过。标准闭环与探针结果要结合分析：优先核对自然语言选择、arm 前 8 行、gripper 和重复底噪。人工 K 权重交换尚无真实物体身份保证；整条 lane 置零是上下文依赖的非线性干预，不据此计算独立贡献百分比或直接修改 P3/bridge。

外部测量源码、helper 快照和结果保存在上述实验的 `probes/` 中；不修改已被两个接续任务锁定的生产/探针源码。分割原始数据与图像不纳入仓库文档。

## 18. 2026-10-06：六份历史正式 checkpoint 的 K 空间读取

用户澄清要看历史版本，而非 step 64 短跑。本次补测六份已完成正式训练的 best.pt，均保存 epoch 1 / step 11012；训练 batch size、初始化和配置不同，因此只作行为描述，不是受控的修复收益实验。为避免用新网络解释旧权重，每份 checkpoint 使用其 git commit 的独立源码快照，逐文件 SHA-256 与 checkpoint.identity.source.files 全部核验通过。

输入统一为已保存的 standard NPZ、蓝块向右指令、与既有探针一致的重复帧/四步 executed-world history。调用各版本自身的 act_with_input，截获采样前输入，再执行原 deployment_cache；G producer 和 per-camera 条件读取均来自该版本原函数。没有在最新生产 checkout 中切换源码，也没有改权重。物块 mask 沿用第 17 节逐像素匹配的模拟器分割。

| checkpoint source | 历史阶段 | 上方相机 K 两两平均重叠 | 手腕相机 K 两两平均重叠 |
|---|---|---:|---:|
| 617fc567 | 初期优化 bs8 | 95.17% | 93.19% |
| 18750208 | bs4 r1 | 62.81% | 53.05% |
| f405c877 | W/global 合并修复 | 92.26% | 94.20% |
| 4f0a5b9b | identity/trajectory 修复 | 95.04% | 86.51% |
| eb9e62fb | deep-repair r2 | 84.59% | 85.19% |
| d931fe01 | S-interval repair 正式长跑 | 41.73% | 58.85% |

重叠定义为每相机六对 K 的 sum(min(p_i,p_j)) 均值，100% 表示同一空间读分布；它不是语义向量 cosine、对象识别率或独立实体数。早期多份权重存在强空间重复；d931fe01 分布分离增大，仍不能据此认定一槽一物。

d931fe01 的 top 像素重心依次为 (88.5,120.8)、(53.5,116.5)、(37.9,106.2)、(104.0,21.6)，wrist 为 (50.3,25.8)、(41.9,22.4)、(42.9,51.0)、(57.9,50.4)。K1 读较宽桌面操作带，K2/K3 的 top 读分布重叠 79.11%，K4 偏上方背板边缘；腕部 K3/K4 有大量下方背景质量。实际可见红/蓝物块上的读质量：top 各 K 为 0.564/0.352/0.136/0.911%，wrist 为 5.748/5.245/1.580/2.526%。此结果是采样位置对可见 mask 的概率质量；上下文视觉特征仍可能携带区域外物体信息，不能把该量直接称为语义信息占比或定位准确率。

同一蓝右指令下，4f0a5b9b/eb9e62fb 的 S real mass 给 K3 约 99.80/99.99%；d931fe01 为 K1–K4 19.39/13.70/42.52/10.12%（余量为 null）。只有一条指令，不能由此宣称所有指令均选 K3。单画面也没有验证跨视角语义身份或跨时间跟踪。

证据根目录：`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/probes/historical-k-slots-20261006-v1/`。manifest.json 固定 checkpoint/source，probe.py 为测量快照，六个提交目录各含 summary/分布 NPZ/图，comparison.json 为紧凑比较，training-audit.json 已核对六个完整 epoch 记录。复现参数保存为各提交.command.json；应在 manifest 中对应源码目录、GPU5、PYTHONPATH=该目录执行，输出必须使用新目录。仓库只存本节统计，不保存源码快照、模型或图像。
